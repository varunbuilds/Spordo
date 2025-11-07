import { AppSidebar } from "../components/app-sidebar";
import { SiteHeader } from "../components/site-header";
import { SidebarInset, SidebarProvider } from "../components/ui/sidebar";
import { useEffect, useRef, useState } from "react";
import { Pose } from "@mediapipe/pose";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Button } from "../components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";

const SportsCoach = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [sport, setSport] = useState("cricket");
  const [cameraActive, setCameraActive] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [formScore, setFormScore] = useState(0);
  const [phase, setPhase] = useState("");
  const [shotName, setShotName] = useState("");
  const [shotConfidence, setShotConfidence] = useState(0);
  const [topShots, setTopShots] = useState([]);
  const [allFeedback, setAllFeedback] = useState([]);
  const ws = useRef(null);
  const updateTimeoutRef = useRef(null);
  const lastUpdateRef = useRef({
    feedback: "",
    formScore: 0,
    phase: "",
    shotName: "",
    shotConfidence: 0,
  });

  useEffect(() => {
    // Establish WebSocket connection when camera turns on
    if (cameraActive) {
      ws.current = new WebSocket("ws://localhost:8000/ws");
      ws.current.onopen = () => console.log("WebSocket connected!");
      ws.current.onclose = () => console.log("WebSocket disconnected.");
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "coaching_feedback") {
          // Clear previous timeout
          if (updateTimeoutRef.current) {
            clearTimeout(updateTimeoutRef.current);
          }

          // Debounce updates - only update if there's a significant change or after 200ms
          updateTimeoutRef.current = setTimeout(() => {
            const lastUpdate = lastUpdateRef.current;

            // Only update if values have changed significantly
            if (
              data.message !== lastUpdate.feedback ||
              Math.abs((data.form_score || 0) - lastUpdate.formScore) > 2 ||
              data.phase !== lastUpdate.phase ||
              data.shot_name !== lastUpdate.shotName ||
              Math.abs(
                (data.shot_confidence || 0) - lastUpdate.shotConfidence
              ) > 0.05
            ) {
              // Update Gemini tip only when it's not empty
              if (data.message) {
                setFeedback(data.message);
              }
              setFormScore(data.form_score || 0);
              setPhase(data.phase || "");
              setShotName(data.shot_name || "");
              setShotConfidence(data.shot_confidence || 0);
              setTopShots(data.top_shots || []);
              setAllFeedback(data.all_feedback || []);

              // Update last values
              lastUpdateRef.current = {
                feedback: data.message || "",
                formScore: data.form_score || 0,
                phase: data.phase || "",
                shotName: data.shot_name || "",
                shotConfidence: data.shot_confidence || 0,
              };
            }
          }, 200); // 200ms debounce
        }
      };
    } else {
      ws.current?.close();
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    }

    // Cleanup on component unmount
    return () => {
      ws.current?.close();
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, [cameraActive]);

  useEffect(() => {
    const pose = new Pose({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 0,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      if (!canvasRef.current) return;
      const canvasCtx = canvasRef.current.getContext("2d");
      canvasCtx.save();
      canvasCtx.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );
      canvasCtx.drawImage(
        results.image,
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );

      if (results.poseLandmarks) {
        // Draw pose landmarks and connections only
        drawConnectors(
          canvasCtx,
          results.poseLandmarks,
          Pose.POSE_CONNECTIONS,
          {
            color: "#00FF00",
            lineWidth: 4,
          }
        );
        drawLandmarks(canvasCtx, results.poseLandmarks, {
          color: "#FF0000",
          lineWidth: 2,
          radius: 2.5,
        });
        // All feedback information is shown in the sidebar panel
      }
      canvasCtx.restore();
    });

    let camera = null;
    if (cameraActive && videoRef.current) {
      camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current) {
            await pose.send({ image: videoRef.current });
            if (ws.current && ws.current.readyState === WebSocket.OPEN) {
              const canvas = canvasRef.current;
              // Send a compressed JPEG image as a Base64 string
              ws.current.send(canvas.toDataURL("image/jpeg", 0.7));
            }
          }
        },
        width: 1280,
        height: 720,
      });
      camera.start();
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
      if (canvasRef.current) {
        const canvasCtx = canvasRef.current.getContext("2d");
        canvasCtx.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );
      }
    }

    return () => {
      if (camera) {
        camera.stop();
      }
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
      pose.close();
    };
  }, [sport, cameraActive]);

  return (
    <SidebarProvider
      style={{
        "--sidebar-width": "calc(var(--spacing) * 72)",
        "--header-height": "calc(var(--spacing) * 12)",
      }}
    >
      <AppSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader title="Sports Coach" />
        <div className="flex flex-1 flex-col p-4 md:p-6">
          {/* Controls */}
          <div className="flex items-center gap-4 mb-4">
            <Select value={sport} onValueChange={setSport}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select a sport" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="cricket">Cricket</SelectItem>
                <SelectItem value="basketball">Basketball</SelectItem>
                <SelectItem value="fitness">Fitness</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={() => setCameraActive(!cameraActive)}>
              {cameraActive ? "Turn Off Camera" : "Turn On Camera"}
            </Button>
          </div>

          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1">
            {/* Left Side - Video Camera */}
            <Card className="flex flex-col">
              <CardHeader>
                <CardTitle>Live Video</CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <div className="relative w-full border aspect-video">
                  <video
                    ref={videoRef}
                    className="rounded-2xl shadow-lg absolute inset-0 w-full h-full"
                    style={{ display: "none" }}
                  />
                  <canvas
                    ref={canvasRef}
                    className="rounded-2xl shadow-lg absolute inset-0 w-full h-full"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Right Side - Text Feedback */}
            <Card className="flex flex-col">
              <CardHeader>
                <CardTitle>Cricket Coaching Feedback</CardTitle>
              </CardHeader>
              <CardContent className="flex-1">
                <div className="h-full rounded-lg overflow-y-auto space-y-4">
                  {cameraActive ? (
                    <>
                      {/* Form Score */}
                      {formScore > 0 && (
                        <div className="bg-gradient-to-br from-chart-3/20 to-chart-5/10 border border-border rounded-lg p-5 shadow-sm transition-all duration-300">
                          <div className="flex justify-between items-center mb-3">
                            <span className="text-sm font-medium text-muted-foreground">
                              Form Score
                            </span>
                            <span className="text-3xl font-bold tabular-nums text-chart-3 transition-all duration-300">
                              {formScore}/100
                            </span>
                          </div>
                          <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden">
                            <div
                              className="bg-gradient-to-r from-chart-3 to-chart-1 h-full rounded-full transition-all duration-700 ease-out"
                              style={{ width: `${formScore}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Phase Badge */}
                      {phase && (
                        <div className="flex items-center gap-2 transition-all duration-300">
                          <div className="bg-accent border border-border px-4 py-2 rounded-lg font-medium text-sm text-foreground inline-flex items-center gap-2 transition-all duration-300">
                            <span>📍</span>
                            <span>
                              Phase:{" "}
                              {phase.charAt(0).toUpperCase() + phase.slice(1)}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Shot Classification - ML Prediction */}
                      {shotName &&
                        shotName !== "N/A" &&
                        shotName !== "Analyzing..." && (
                          <div className="bg-gradient-to-br from-chart-3/20 to-chart-4/10 border border-chart-3/30 rounded-lg p-5 shadow-sm transition-all duration-300">
                            <div className="flex items-center justify-between mb-4">
                              <div className="flex items-center gap-3">
                                <div className="bg-chart-3/20 rounded-full p-2">
                                  <span className="text-2xl">🏏</span>
                                </div>
                                <div>
                                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                                    Shot Detected
                                  </p>
                                  <p className="text-xl font-bold text-foreground mt-0.5 transition-all duration-300">
                                    {shotName}
                                  </p>
                                </div>
                              </div>
                              <div className="text-right">
                                <p className="text-xs text-muted-foreground uppercase tracking-wide">
                                  Confidence
                                </p>
                                <p className="text-2xl font-bold text-chart-3 tabular-nums transition-all duration-300">
                                  {Math.round(shotConfidence * 100)}%
                                </p>
                              </div>
                            </div>
                            {topShots && topShots.length > 1 && (
                              <div className="pt-3 border-t border-border/50 space-y-2 transition-all duration-300">
                                <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                                  Other possibilities:
                                </p>
                                {topShots
                                  .slice(1, 3)
                                  .map(([shot, conf], idx) => (
                                    <div
                                      key={idx}
                                      className="flex justify-between items-center text-sm transition-all duration-300"
                                    >
                                      <span className="text-foreground/80">
                                        {shot}
                                      </span>
                                      <span className="text-chart-2 font-semibold tabular-nums">
                                        {Math.round(conf * 100)}%
                                      </span>
                                    </div>
                                  ))}
                              </div>
                            )}
                          </div>
                        )}

                      {/* Main Coaching Tip */}
                      {feedback && (
                        <div className="bg-gradient-to-br from-amber-500/10 to-amber-600/5 border border-amber-500/30 rounded-lg p-4 shadow-sm transition-all duration-300">
                          <div className="flex items-start gap-3">
                            <div className="bg-amber-500/20 rounded-full p-1.5 flex-shrink-0">
                              <span className="text-xl">💡</span>
                            </div>
                            <div className="flex-1">
                              <p className="font-semibold text-foreground mb-1.5 text-sm uppercase tracking-wide">
                                Coach's Tip
                              </p>
                              <p className="text-foreground/90 text-sm leading-relaxed transition-all duration-300">
                                {feedback}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Detailed Feedback */}
                      {allFeedback.length > 0 && (
                        <div className="space-y-3">
                          <div className="flex items-center gap-2 px-1">
                            <span className="text-lg">⚠️</span>
                            <p className="text-sm font-semibold text-foreground uppercase tracking-wide">
                              Detailed Analysis
                            </p>
                          </div>
                          <div className="space-y-2">
                            {allFeedback.map((item, idx) => (
                              <div
                                key={idx}
                                className={`p-3 rounded-lg border-l-4 shadow-xs transition-all duration-300 ${
                                  item.status === "error"
                                    ? "bg-destructive/10 border-destructive"
                                    : item.status === "warning"
                                    ? "bg-amber-500/10 border-amber-500"
                                    : "bg-green-500/10 border-green-500"
                                }`}
                              >
                                <div className="flex items-start gap-2.5">
                                  <span className="text-base flex-shrink-0 mt-0.5">
                                    {item.status === "error"
                                      ? "❌"
                                      : item.status === "warning"
                                      ? "⚠️"
                                      : "✅"}
                                  </span>
                                  <div className="flex-1 min-w-0">
                                    <p
                                      className={`text-sm font-medium leading-snug ${
                                        item.status === "error"
                                          ? "text-destructive"
                                          : item.status === "warning"
                                          ? "text-amber-400"
                                          : "text-green-400"
                                      }`}
                                    >
                                      {item.message}
                                    </p>
                                    {item.metric && (
                                      <p className="text-xs text-muted-foreground mt-1.5">
                                        {item.metric}
                                      </p>
                                    )}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* No Issues */}
                      {!feedback &&
                        formScore === 0 &&
                        allFeedback.length === 0 && (
                          <div className="text-center py-12 text-muted-foreground">
                            <div className="bg-accent/50 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                              <p className="text-3xl">👀</p>
                            </div>
                            <p className="text-base font-medium text-foreground mb-2">
                              Analyzing your batting form...
                            </p>
                            <p className="text-sm">
                              Get into your batting stance for feedback
                            </p>
                          </div>
                        )}

                      {/* Invalid Stance Warning */}
                      {formScore === 0 && allFeedback.length > 0 && (
                        <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-4 shadow-sm">
                          <div className="flex items-start gap-3">
                            <div className="bg-destructive/20 rounded-full p-1.5 flex-shrink-0">
                              <span className="text-xl">🚫</span>
                            </div>
                            <div className="flex-1">
                              <p className="font-semibold text-destructive mb-1.5 text-sm uppercase tracking-wide">
                                Invalid Position
                              </p>
                              <p className="text-foreground/90 text-sm leading-relaxed">
                                {allFeedback[0]?.message ||
                                  "Please stand up in batting stance"}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      <div className="bg-accent/50 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                        <p className="text-3xl">📹</p>
                      </div>
                      <p className="text-base font-medium text-foreground mb-2">
                        Camera Inactive
                      </p>
                      <p className="text-sm">
                        Turn on the camera to start receiving feedback
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
};

export default SportsCoach;
