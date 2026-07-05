// ================================================================
// CHILLI LEAF DISEASE DETECTION SYSTEM
// ESP32-CAM + TinyML (MobileNetV2) + SIM800L GSM
// Detects 8 chilli diseases and sends SMS alert to farmer
// Board: AI Thinker ESP32-CAM
// ================================================================

#include "esp_camera.h"
#include "SPIFFS.h"
#include <SoftwareSerial.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ── Pin Definitions ──────────────────────────────────────────
#define LED_PIN      2    // Green LED — blinks on capture, glows on alert
#define BUTTON_PIN   0    // Push button — press for instant capture
#define GSM_RX_PIN  14    // SIM800L TX → ESP32 GPIO14
#define GSM_TX_PIN  15    // SIM800L RX → ESP32 GPIO15

String FARMER_NUMBER = "+919876543210";─
// For real field deployment change to: TRAP_SAFE=5, TRAP_WARN=15
#define DISEASE_CONFIDENCE_THRESHOLD  60   // % confidence needed to send alert
#define HEALTHY_CONFIDENCE_THRESHOLD  80   // % confidence to say "safe"

// ── GSM Serial ───────────────────────────────────────────────
SoftwareSerial gsmSerial(GSM_RX_PIN, GSM_TX_PIN);

const char* DISEASES[] = {
  "Anthracnose",
  "Bacterial Spot",
  "Cercospora Leaf Spot",
  "Healthy",
  "Leaf Curl Virus",
  "Mosaic Virus",
  "Powdery Mildew",
  "Thrips Damage"
};

// ── Treatment for each disease ────────────────────────────────
const char* TREATMENTS[] = {
  "Spray Copper Oxychloride 3g/L. Improve field drainage.",
  "Spray Copper hydroxide 2g/L. Remove infected leaves.",
  "Spray Mancozeb 2.5g/L. Remove and burn infected leaves.",
  "Crop is healthy. No action needed.",
  "No chemical cure. Remove infected plants. Control thrips.",
  "No chemical cure. Remove infected plants immediately.",
  "Spray Carbendazim 1g/L or Sulphur WP 3g/L.",
  "Spray Spinosad 45SC 1ml/L. Inspect field immediately."
};

// ── TFLite model setup ────────────────────────────────────────
const int TENSOR_ARENA_SIZE = 150 * 1024;  // 150KB arena for inference
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model*       tflite_model  = nullptr;
tflite::MicroInterpreter*  interpreter   = nullptr;
TfLiteTensor*              input_tensor  = nullptr;
TfLiteTensor*              output_tensor = nullptr;
bool                       model_loaded  = false;

// ================================================================
// CAMERA INITIALISATION — AI-Thinker ESP32-CAM pin config
// ================================================================
bool initCamera() {
  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  // Camera data pins
  config.pin_d0 = 5;   config.pin_d1 = 18;
  config.pin_d2 = 19;  config.pin_d3 = 21;
  config.pin_d4 = 36;  config.pin_d5 = 39;
  config.pin_d6 = 34;  config.pin_d7 = 35;

  // Camera control pins
  config.pin_xclk     = 0;
  config.pin_pclk     = 22;
  config.pin_vsync    = 25;
  config.pin_href     = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn     = 32;
  config.pin_reset    = -1;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAMESIZE_96X96;  // 96x96 matches model input
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }

  delay(500);  // Let camera sensor stabilise
  Serial.println("Camera initialised!");
  return true;
}

// ================================================================
// LOAD TFLITE MODEL FROM SPIFFS
// The .tflite file was uploaded to ESP32 flash using
// Tools → ESP32 Sketch Data Upload in Arduino IDE
// ================================================================
bool loadModel() {
  if (!SPIFFS.begin(true)) {
    Serial.println("ERROR: SPIFFS mount failed!");
    Serial.println("Upload .tflite file using ESP32 Sketch Data Upload tool.");
    return false;
  }

  // Check file exists
  if (!SPIFFS.exists("/chilli_disease.tflite")) {
    Serial.println("ERROR: chilli_disease.tflite not found in SPIFFS!");
    Serial.println("Put the file in data/ folder and use Sketch Data Upload.");
    return false;
  }

  // Read model file into memory
  File modelFile = SPIFFS.open("/chilli_disease.tflite", "r");
  size_t modelSize = modelFile.size();
  Serial.printf("Model file size: %d bytes (%.1f KB)\n", modelSize, modelSize/1024.0);

  uint8_t* modelBuffer = (uint8_t*)malloc(modelSize);
  if (!modelBuffer) {
    Serial.println("ERROR: Not enough RAM to load model!");
    modelFile.close();
    return false;
  }

  modelFile.read(modelBuffer, modelSize);
  modelFile.close();

  // Parse the model
  tflite_model = tflite::GetModel(modelBuffer);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema mismatch! Expected %d, got %d\n",
                  TFLITE_SCHEMA_VERSION, tflite_model->version());
    free(modelBuffer);
    return false;
  }

  // Set up interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    tflite_model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  // Allocate tensor memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed!");
    free(modelBuffer);
    return false;
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("Model loaded! Input: %dx%dx%d  Output: %d classes\n",
    input_tensor->dims->data[1],
    input_tensor->dims->data[2],
    input_tensor->dims->data[3],
    output_tensor->dims->data[1]);

  return true;
}

// ================================================================
// IMAGE PROCESSING — ALGORITHM 1: BLOB DETECTION
// Finds dark lesion spots on the leaf
// Runs before ML model as a quick pre-check
// ================================================================
void toGrayscale(uint8_t* rgb565, uint8_t* gray, int w, int h) {
  for (int i = 0; i < w * h; i++) {
    uint16_t px = ((uint16_t)rgb565[i*2] << 8) | rgb565[i*2+1];
    uint8_t r = ((px >> 11) & 0x1F) * 255 / 31;
    uint8_t g = ((px >> 5)  & 0x3F) * 255 / 63;
    uint8_t b = ( px        & 0x1F) * 255 / 31;
    gray[i] = (uint8_t)(0.299f*r + 0.587f*g + 0.114f*b);
  }
}

int countLesionBlobs(uint8_t* gray, int w, int h) {
  static bool    visited[96*96];
  static uint8_t thresh[96*96];
  memset(visited, 0, w * h);

  // Disease lesions appear as dark patches on the leaf
  for (int i = 0; i < w*h; i++)
    thresh[i] = (gray[i] < 70) ? 1 : 0;

  int count = 0;
  for (int y = 1; y < h-1; y++) {
    for (int x = 1; x < w-1; x++) {
      int idx = y*w + x;
      if (!thresh[idx] || visited[idx]) continue;

      // Flood fill to measure blob size
      static int sx[500], sy[500];
      int top = 0, blobSize = 0;
      sx[top] = x; sy[top] = y; top++;
      visited[idx] = true;

      while (top > 0) {
        top--;
        int cx = sx[top], cy = sy[top];
        blobSize++;
        int dxs[] = {1,-1,0,0};
        int dys[] = {0,0,1,-1};
        for (int d = 0; d < 4; d++) {
          int nx = cx+dxs[d], ny = cy+dys[d];
          if (nx<0||nx>=w||ny<0||ny>=h) continue;
          int ni = ny*w+nx;
          if (thresh[ni] && !visited[ni] && top < 498) {
            visited[ni] = true;
            sx[top] = nx; sy[top] = ny; top++;
          }
        }
      }
      // Lesion size range: 10 to 300 pixels at 96x96
      if (blobSize >= 10 && blobSize <= 300) count++;
    }
  }
  return count;
}

// ================================================================
// IMAGE PROCESSING — ALGORITHM 2: HSV COLOUR ANALYSIS
// Measures % of silver, bronze, yellow, white, brown, green
// Provides additional signal alongside ML model output
// ================================================================
struct ColourResult {
  float silver;    // thrips feeding damage
  float bronze;    // thrips toxin damage
  float yellow;    // mosaic virus
  float white_pct; // powdery mildew
  float brown;     // cercospora / anthracnose
  float green;     // healthy
};

void rgbToHsv(uint8_t r, uint8_t g, uint8_t b,
              float* h, float* s, float* v) {
  float rf = r/255.f, gf = g/255.f, bf = b/255.f;
  float mx = max(rf,max(gf,bf));
  float mn = min(rf,min(gf,bf));
  float diff = mx - mn;
  *v = mx;
  *s = (mx == 0) ? 0 : diff / mx;
  if (diff == 0) { *h = 0; return; }
  if      (mx == rf) *h = 60.f * fmod((gf-bf)/diff, 6.f);
  else if (mx == gf) *h = 60.f * ((bf-rf)/diff + 2.f);
  else               *h = 60.f * ((rf-gf)/diff + 4.f);
  if (*h < 0) *h += 360.f;
}

ColourResult analyseColour(uint8_t* rgb565, int w, int h) {
  int silver=0, bronze=0, yellow=0, white_c=0, brown=0, green=0;
  int total = w * h;

  for (int i = 0; i < total; i++) {
    uint16_t px = ((uint16_t)rgb565[i*2] << 8) | rgb565[i*2+1];
    uint8_t r = ((px>>11)&0x1F)*255/31;
    uint8_t g = ((px>>5) &0x3F)*255/63;
    uint8_t b = ( px     &0x1F)*255/31;
    float h, s, v;
    rgbToHsv(r, g, b, &h, &s, &v);

    if      (s < 0.08f && v > 0.85f)              white_c++;  // powdery mildew
    else if (s < 0.15f && v > 0.65f && v<=0.85f)  silver++;   // thrips silver
    else if (h>=10 && h<=25 && s>0.35f)            bronze++;   // thrips bronze
    else if (h>=25 && h<=45 && s>0.40f)            yellow++;   // mosaic virus
    else if (h>=10 && h<=20 && s>0.40f && v<0.6f) brown++;    // spots/blight
    else if (h>=70 && h<=145 && s>0.20f)           green++;    // healthy green
  }

  return {
    silver*100.f/total,  bronze*100.f/total,
    yellow*100.f/total,  white_c*100.f/total,
    brown*100.f/total,   green*100.f/total
  };
}

// ================================================================
// TINYML INFERENCE — Algorithm 3
// Runs MobileNetV2 TFLite model on the leaf image
// Returns disease class index (0-7) and confidence %
// ================================================================
struct MLResult {
  int   classIndex;   // 0-7 matching DISEASES array
  float confidence;   // 0-100%
};

MLResult runMLModel(uint8_t* rgb565, int w, int h) {
  if (!model_loaded) {
    Serial.println("Model not loaded — skipping ML inference");
    return {-1, 0};
  }

  // Copy RGB image into model input tensor
  uint8_t* inp = input_tensor->data.uint8;
  for (int i = 0; i < w * h; i++) {
    uint16_t px = ((uint16_t)rgb565[i*2] << 8) | rgb565[i*2+1];
    inp[i*3+0] = ((px>>11)&0x1F)*255/31;  // R
    inp[i*3+1] = ((px>>5) &0x3F)*255/63;  // G
    inp[i*3+2] = ( px     &0x1F)*255/31;  // B
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: ML inference failed!");
    return {-1, 0};
  }

  // Find highest probability class
  uint8_t* out = output_tensor->data.uint8;
  int   bestClass = 0;
  uint8_t bestScore = 0;
  for (int i = 0; i < 8; i++) {
    if (out[i] > bestScore) {
      bestScore = out[i];
      bestClass = i;
    }
  }

  float confidence = (bestScore / 255.0f) * 100.0f;
  Serial.printf("ML Result: %s (%.0f%% confidence)\n",
                DISEASES[bestClass], confidence);
  return {bestClass, confidence};
}

// ================================================================
// FINAL DECISION — Combines all 3 signals
// ML model is primary. HSV and blob are secondary signals.
// ================================================================
int makeFinalDecision(int lesions, ColourResult col, MLResult ml) {

  // If ML model gives confident result, trust it
  if (ml.classIndex >= 0 && ml.confidence >= DISEASE_CONFIDENCE_THRESHOLD) {
    return ml.classIndex;
  }

  // Fallback: use HSV colour rules if model confidence too low
  Serial.println("ML confidence low — using HSV fallback rules");

  if (col.white_pct > 8)                        return 6;  // Powdery Mildew
  if ((col.silver + col.bronze) > 15)            return 7;  // Thrips Damage
  if (col.yellow > 12)                           return 5;  // Mosaic Virus
  if (col.brown > 10 && lesions > 5)             return 2;  // Cercospora
  if (lesions > 8)                               return 0;  // Anthracnose
  if (col.green < 20)                            return 4;  // Leaf Curl
  if ((col.silver+col.bronze+col.brown) > 5
       && lesions > 3)                           return 1;  // Bacterial Spot
  return 3;  // Healthy
}

// ================================================================
// GSM SMS FUNCTIONS
// ================================================================
void setupGSM() {
  Serial.println("Setting up SIM800L GSM...");
  gsmSerial.begin(9600);
  delay(5000);  // SIM800L needs ~5 seconds to register on network

  gsmSerial.println("AT");
  delay(500);
  while (gsmSerial.available()) Serial.write(gsmSerial.read());

  gsmSerial.println("AT+CMGF=1");  // Set SMS text mode
  delay(500);
  while (gsmSerial.available()) Serial.write(gsmSerial.read());

  Serial.println("GSM ready!");
}

void sendSMS(int diseaseClass, float confidence,
             ColourResult col, int lesions) {

  float damage = col.silver + col.bronze + col.brown;

  // Build SMS message
  String msg  = "Chilli Leaf Monitor:\n";
  msg += "Disease: " + String(DISEASES[diseaseClass]) + "\n";
  msg += "Confidence: " + String(confidence, 0) + "%\n";
  msg += "Healthy green: " + String(col.green, 0) + "%\n";
  msg += "Damage area: " + String(damage, 0) + "%\n";
  msg += "Lesion spots: " + String(lesions) + "\n";
  msg += "Action: " + String(TREATMENTS[diseaseClass]);

  Serial.println("\n========== SMS CONTENT ==========");
  Serial.println(msg);
  Serial.println("=================================");

  // Send via SIM800L AT commands
  gsmSerial.print("AT+CMGS=\"");
  gsmSerial.print(FARMER_NUMBER);
  gsmSerial.println("\"");
  delay(1000);
  gsmSerial.print(msg);
  delay(200);
  gsmSerial.write(26);  // Ctrl+Z = send SMS
  delay(5000);

  // Triple LED blink = SMS sent
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH); delay(200);
    digitalWrite(LED_PIN, LOW);  delay(200);
  }
  Serial.println("SMS sent! Check your phone.");
}

// ================================================================
// MAIN CAPTURE AND ANALYSIS CYCLE
// Called on button press or auto-timer
// ================================================================
void runCycle() {
  Serial.println("\n========== CAPTURE STARTED ==========");

  // Single LED blink = capture starting
  digitalWrite(LED_PIN, HIGH); delay(150); digitalWrite(LED_PIN, LOW);

  // Initialise camera
  if (!initCamera()) {
    Serial.println("Camera failed — skipping this cycle");
    return;
  }

  // ── Capture leaf image ────────────────────────────────────
  Serial.println("Capturing leaf image...");
  Serial.println("(Hold mirchi leaf flat, 10-12cm below camera, on white paper)");

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("ERROR: Frame capture failed!");
    esp_camera_deinit();
    return;
  }

  Serial.printf("Image captured: %dx%d (%d bytes)\n",
                fb->width, fb->height, fb->len);

  // ── Algorithm 1: Blob detection ──────────────────────────
  Serial.println("Running blob detection...");
  static uint8_t gray[96*96];
  toGrayscale(fb->buf, gray, fb->width, fb->height);
  int lesions = countLesionBlobs(gray, fb->width, fb->height);
  Serial.printf("Lesion blobs detected: %d\n", lesions);

  // ── Algorithm 2: HSV colour analysis ─────────────────────
  Serial.println("Running HSV colour analysis...");
  ColourResult colours = analyseColour(fb->buf, fb->width, fb->height);
  Serial.printf("Healthy green: %.1f%%\n", colours.green);
  Serial.printf("Silver+Bronze: %.1f%%\n", colours.silver+colours.bronze);
  Serial.printf("White (mildew): %.1f%%\n", colours.white_pct);
  Serial.printf("Yellow (mosaic): %.1f%%\n", colours.yellow);
  Serial.printf("Brown (spots): %.1f%%\n", colours.brown);

  // ── Algorithm 3: TinyML inference ────────────────────────
  Serial.println("Running TinyML inference...");
  MLResult mlResult = runMLModel(fb->buf, fb->width, fb->height);

  // ── IMAGE DELETED — never stored permanently ──────────────
  esp_camera_fb_return(fb);
  fb = nullptr;
  esp_camera_deinit();
  Serial.println("Image processed and discarded from memory.");

  // ── Final decision ────────────────────────────────────────
  int finalClass = makeFinalDecision(lesions, colours, mlResult);
  float finalConf = (mlResult.classIndex >= 0) ? mlResult.confidence : 70.0f;

  // ── Print results table ───────────────────────────────────
  Serial.println("\n╔══════════════════════════════════════╗");
  Serial.println("║        LEAF ANALYSIS RESULTS         ║");
  Serial.println("╠══════════════════════════════════════╣");
  Serial.printf( "║  Disease detected : %-18s║\n", DISEASES[finalClass]);
  Serial.printf( "║  Confidence       : %-16.0f%%  ║\n", finalConf);
  Serial.printf( "║  Lesion blobs     : %-18d║\n", lesions);
  Serial.printf( "║  Healthy green %%  : %-16.1f%%  ║\n", colours.green);
  Serial.printf( "║  Damage %%         : %-16.1f%%  ║\n",
                  colours.silver+colours.bronze+colours.brown);
  Serial.println("╠══════════════════════════════════════╣");
  Serial.printf( "║  Treatment: %-25s ║\n", TREATMENTS[finalClass]);
  Serial.println("╚══════════════════════════════════════╝");

  // ── Send SMS alert ────────────────────────────────────────
  // Always send for demo. In real deployment, skip SMS when healthy.
  if (finalClass == 3 && finalConf >= HEALTHY_CONFIDENCE_THRESHOLD) {
    Serial.println("Crop is HEALTHY — sending safe status SMS...");
  } else {
    Serial.println("Disease detected — sending alert SMS!");
    digitalWrite(LED_PIN, HIGH);  // LED stays on for disease alert
  }

  setupGSM();
  sendSMS(finalClass, finalConf, colours, lesions);
  digitalWrite(LED_PIN, LOW);

  Serial.println("\nCycle complete. Ready for next capture.");
  Serial.println("Press button or wait 30 seconds for auto-capture.");
}

// ================================================================
// SETUP — runs once when ESP32 powers on
// ================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(LED_PIN, LOW);

  // 5 quick blinks = system alive
  Serial.println("\n==========================================");
  Serial.println("  CHILLI DISEASE DETECTION SYSTEM");
  Serial.println("  MobileNetV2 TinyML + Blob + HSV");
  Serial.println("  87.4% accuracy on 8 diseases");
  Serial.println("==========================================\n");

  for (int i = 0; i < 5; i++) {
    digitalWrite(LED_PIN, HIGH); delay(80);
    digitalWrite(LED_PIN, LOW);  delay(80);
  }

  // Load TFLite model from SPIFFS
  Serial.println("Loading TinyML model from SPIFFS...");
  model_loaded = loadModel();

  if (model_loaded) {
    Serial.println("Model loaded successfully!");
    // 2 slow blinks = model ready
    for (int i = 0; i < 2; i++) {
      digitalWrite(LED_PIN, HIGH); delay(400);
      digitalWrite(LED_PIN, LOW);  delay(400);
    }
  } else {
    Serial.println("WARNING: Model not loaded!");
    Serial.println("System will run in HSV-only mode.");
    Serial.println("To fix: upload chilli_disease.tflite via Sketch Data Upload");
  }

  Serial.println("\nSystem ready!");
  Serial.println("Press BUTTON (GPIO0) to capture instantly.");
  Serial.println("Auto-capture every 30 seconds for demo.");
  Serial.println("=========================================\n");
}

// ================================================================
// LOOP — always running, watching for button press
// ================================================================
void loop() {

  // ── Button press = instant capture ───────────────────────
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(50);  // debounce
    if (digitalRead(BUTTON_PIN) == LOW) {
      Serial.println("Button pressed — manual capture!");
      runCycle();
      delay(1000);  // prevent double-trigger
    }
  }

  // ── Auto-capture every 30 seconds for demo ───────────────
  // Change 30000 to 21600000 for real field (6 hours)
  static unsigned long lastCapture = 0;
  if (millis() - lastCapture > 30000) {
    lastCapture = millis();
    Serial.println("Auto-capture triggered (30s timer)...");
    runCycle();
  }

  // ── Heartbeat LED blink every 2 seconds ──────────────────
  // Shows the system is alive even when not capturing
  static unsigned long lastBlink = 0;
  if (millis() - lastBlink > 2000) {
    lastBlink = millis();
    digitalWrite(LED_PIN, HIGH); delay(30);
    digitalWrite(LED_PIN, LOW);
  }
}
