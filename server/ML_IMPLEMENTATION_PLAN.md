# 🏏 ML-Based Cricket Shot Classification Implementation

## 📊 **Dataset: CricShot10**

**Source:** https://github.com/ascuet/CricShot10  
**Paper:** "CricShotClassify: An approach to classifying batting shots from cricket videos using a convolutional neural network and gated recurrent unit" (Sensors, 2021)

### **10 Cricket Shots Included:**

1. **Cover Drive** - Classic front foot shot through covers
2. **Defense** - Defensive block shot
3. **Flick** - Wrist shot to leg side
4. **Hook** - Short ball shot to leg side
5. **Late Cut** - Delicate shot behind wicket
6. **Lofted** - Aerial shot over fielders
7. **Pull** - Horizontal bat shot to leg side
8. **Square Cut** - Shot square of wicket on off side
9. **Straight Drive** - Shot straight down ground
10. **Sweep** - Shot played from knee

---

## 🚀 **Quick Implementation Strategy**

### **Option A: Use Their Pre-trained Model (FASTEST)**

✅ **Time:** 1-2 hours  
✅ **Accuracy:** Research-validated (~85-90%)  
✅ **Dataset:** Included (request from authors)

**Steps:**

1. Email authors for dataset access (aniksen.cuet09@gmail.com)
2. Download their CNN-GRU model weights
3. Integrate with your MediaPipe pose pipeline
4. Deploy to your app

### **Option B: Transfer Learning (RECOMMENDED)**

✅ **Time:** 3-4 hours  
✅ **Accuracy:** High with fine-tuning  
✅ **Dataset:** Public alternatives available

**Steps:**

1. Use **MoViNet** or **X3D** pre-trained on Kinetics-400
2. Fine-tune on cricket pose sequences
3. Extract pose with MediaPipe → Feed to model
4. Real-time inference in your app

### **Option C: Use Public Pre-trained Model**

✅ **Time:** 2-3 hours  
✅ **Accuracy:** Good for demo  
✅ **Dataset:** Not needed initially

**Steps:**

1. Use **TimeSformer** or **VideoMAE** from Hugging Face
2. Zero-shot classification with cricket shot labels
3. No training required - immediate deployment

---

## 💡 **BEST APPROACH FOR YOU**

I recommend **Option B (Transfer Learning)** because:

- ✅ No need to wait for dataset access
- ✅ Can use publicly available cricket videos
- ✅ Model will be custom to your needs
- ✅ Can demo immediately with existing videos
- ✅ Better control over accuracy

---

## 🛠️ **Implementation Architecture**

```
Webcam Frame
    ↓
MediaPipe Pose Detection (33 landmarks)
    ↓
Pose Sequence Buffer (30 frames = 1 sec)
    ↓
LSTM/Transformer Model
    ↓
Shot Classification
    [Cover Drive: 0.85, Pull: 0.10, Defense: 0.05]
    ↓
Biomechanics Analysis (your existing rules)
    ↓
Gemini AI Coaching Tips
    ↓
WebSocket → Frontend
```

---

## 📦 **What I'll Create for You**

### **Files to Add:**

1. `server/models/cricket_shot_classifier.py` - ML model wrapper
2. `server/models/shot_classifier_lstm.h5` - Trained model weights
3. `server/analysis/ml_cricket_coach.py` - ML + biomechanics hybrid
4. `server/train_shot_classifier.py` - Training script (for reference)
5. `server/download_cricket_videos.py` - Dataset collection script

### **Integration:**

- Replace `enhanced_cricket_coach.py` with `ml_cricket_coach.py`
- Add shot classification to WebSocket response
- Display shot name + confidence in UI

---

## 🎯 **Next Steps**

**Choose Your Path:**

**Path 1: Quick Demo (2 hours)**

- I'll use a pre-trained action recognition model
- Zero-shot classification on 10 cricket shots
- Deploy immediately with existing code

**Path 2: Custom Training (4 hours)**

- I'll download public cricket videos from YouTube
- Extract pose sequences with MediaPipe
- Train LSTM model on your machine
- Deploy custom model

**Path 3: Use CricShot10 (1 week)**

- Email authors for dataset
- Wait for approval
- Use their exact model architecture
- Highest accuracy but slower

---

## ⚡ **My Recommendation: Path 1 (Quick Demo)**

Let me implement this RIGHT NOW:

1. ✅ Download **MoViNet-A0** (pre-trained on Kinetics)
2. ✅ Add cricket shot labels mapping
3. ✅ Integrate with your pose pipeline
4. ✅ Update UI to show shot name
5. ✅ Keep biomechanics feedback for form correction

**Total time:** ~2 hours  
**Accuracy:** Good enough for MVP/demo  
**Can upgrade later:** Yes, retrain with better dataset

---

## 🤔 **Decision Time**

**Reply with:**

- **"Quick Demo"** → I'll implement Path 1 now
- **"Custom Train"** → I'll set up Path 2 with YouTube videos
- **"Wait for CricShot10"** → I'll draft email to authors

Which path do you prefer?
