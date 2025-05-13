let videoElement = document.getElementById('video');
let canvasElement = document.getElementById('outputCanvas');
let feedbackElement = document.getElementById('feedback');
let generateReportButton = document.getElementById('generateReport');
let ctx = canvasElement.getContext('2d');
let poseLandmarker;
let kneeAngles = [];
let timestamps = [];

document.getElementById('videoUpload').addEventListener('change', handleVideoUpload);
generateReportButton.addEventListener('click', generateReport);

async function handleVideoUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  videoElement.src = URL.createObjectURL(file);
  await videoElement.play();

  // Load the MediaPipe Pose Landmarker
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'models/pose_landmarker_lite.task',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
  });

  processVideo();
}

function processVideo() {
  const processFrame = async () => {
    if (videoElement.paused || videoElement.ended) return;

    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    const nowInMs = performance.now();
    const results = await poseLandmarker.detectForVideo(videoElement, nowInMs);

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      drawLandmarks(landmarks);
      const angle = calculateKneeAngle(landmarks);
      kneeAngles.push(angle);
      timestamps.push(videoElement.currentTime);
      provideFeedback(angle);
    }

    requestAnimationFrame(processFrame);
  };

  processFrame();
}

function drawLandmarks(landmarks) {
  ctx.fillStyle = 'red';
  for (let landmark of landmarks) {
    ctx.beginPath();
    ctx.arc(landmark.x * canvasElement.width, landmark.y * canvasElement.height, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function calculateKneeAngle(landmarks) {
  // Left hip: 23, left knee: 25, left ankle: 27
  const hip = landmarks[23];
  const knee = landmarks[25];
  const ankle = landmarks[27];

  const angle = calculateAngle(hip, knee, ankle);
  return angle;
}

function calculateAngle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x ** 2 + ab.y ** 2);
  const magCB = Math.sqrt(cb.x ** 2 + cb.y ** 2);

  const angleRad = Math.acos(dot / (magAB * magCB));
  const angleDeg = (angleRad * 180) / Math.PI;
  return angleDeg;
}

function provideFeedback(angle) {
  let message = '';
  if (angle > 160) {
    message = 'You are standing upright.';
  } else if (angle > 90) {
    message = 'You are squatting. Good depth!';
  } else {
    message = 'You are in a deep squat. Excellent!';
  }
  feedbackElement.textContent = `Knee Angle: ${angle.toFixed(2)}° - ${message}`;
}

function generateReport() {
  // For simplicity, we'll just log the angles and timestamps.
  // In a real application, you might send this data to a backend to generate a PDF.
  console.log('Knee Angles:', kneeAngles);
  console.log('Timestamps:', timestamps);
  alert('Report generation is not implemented in this demo.');
}
