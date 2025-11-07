import Navbar from "@/components/Navbar";
import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Target,
  Activity,
  Lightbulb,
  Video,
  User,
  Zap,
  BarChart3,
} from "lucide-react";

const Home = () => {
  return (
    <>
      <Navbar />
      {/* Hero Section */}
      <section className="w-full min-h-[90vh] flex items-center justify-center  bg-background">
        <div className="grid grid-cols-1 md:grid-cols-2 items-center gap-8 max-w-6xl mx-auto w-full">
          <div>
            <span className="inline-block mb-4 text-xs md:text-sm text-primary font-medium border border-dotted border-primary rounded-full px-3 py-1">
              AI-powered activity
            </span>
            <h3 className="text-4xl md:text-6xl font-semibold text-foreground">
              Your Personal AI Sports Companion
            </h3>
            <p className="text-base md:text-lg text-muted-foreground my-4 md:my-6 max-w-lg">
              Elevate your performance! Get real-time insights, expert coaching,
              and tailored training powered by advanced AI. Stay motivated and
              achieve your goals with ease.
            </p>
            <div className="flex gap-4">
              <Link
                to="/dashboard"
                className="cursor-pointer bg-[var(--chart-3)] text-primary-foreground font-medium py-2 px-4 rounded transition-all hover:bg-primary/90 active:scale-95 flex items-center justify-center"
              >
                Go to Dashboard
              </Link>
              <Link
                to="/gallery"
                className="cursor-pointer bg-secondary text-secondary-foreground font-medium py-2 px-4 rounded transition-all border border-border hover:bg-accent active:scale-95 flex items-center justify-center"
              >
                View Gallery
              </Link>
            </div>
          </div>
          <ShuffleGrid />
        </div>
      </section>

      {/* Features Section */}
      <section className="w-full py-24 px-8 bg-background">
        <div className="max-w-6xl mx-auto">
          <div className="mb-16">
            <h2 className="text-3xl font-semibold text-foreground mb-3">
              Features
            </h2>
            <p className="text-muted-foreground">
              Comprehensive cricket analysis powered by AI
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-md bg-chart-3/10 flex items-center justify-center">
                  <Target className="w-5 h-5 text-chart-3" />
                </div>
                <h3 className="font-medium text-foreground">
                  Real-Time Analysis
                </h3>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Instant feedback on batting form with pose detection analyzing
                33 body landmarks at 20+ FPS.
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-md bg-chart-3/10 flex items-center justify-center">
                  <Activity className="w-5 h-5 text-chart-3" />
                </div>
                <h3 className="font-medium text-foreground">
                  Shot Classification
                </h3>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                AI identifies 10 different cricket shots with confidence scores
                and alternative predictions.
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-md bg-chart-3/10 flex items-center justify-center">
                  <Lightbulb className="w-5 h-5 text-chart-3" />
                </div>
                <h3 className="font-medium text-foreground">AI Coach Tips</h3>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Personalized coaching advice powered by Google Gemini AI with
                actionable insights.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="w-full py-24 px-8 bg-muted/30">
        <div className="max-w-6xl mx-auto">
          <div className="mb-16">
            <h2 className="text-3xl font-semibold text-foreground mb-3">
              How It Works
            </h2>
            <p className="text-muted-foreground">
              Four simple steps to improve your cricket technique
            </p>
          </div>

          <div className="space-y-8">
            {[
              {
                num: "01",
                icon: Video,
                title: "Camera Capture",
                desc: "Turn on your webcam and position yourself in the batting stance",
              },
              {
                num: "02",
                icon: User,
                title: "Pose Detection",
                desc: "MediaPipe extracts 33 body landmarks in real-time with high precision",
              },
              {
                num: "03",
                icon: Zap,
                title: "AI Analysis",
                desc: "Shot classification and biomechanics evaluation running simultaneously",
              },
              {
                num: "04",
                icon: BarChart3,
                title: "Get Feedback",
                desc: "Instant coaching tips, form scores, and detailed analysis on your screen",
              },
            ].map((step, idx) => (
              <div key={idx} className="flex gap-6 items-start group">
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 rounded-lg border border-border bg-card flex items-center justify-center group-hover:border-chart-3 transition-colors">
                    <step.icon className="w-5 h-5 text-muted-foreground group-hover:text-chart-3 transition-colors" />
                  </div>
                </div>
                <div className="space-y-1 pt-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-muted-foreground">
                      {step.num}
                    </span>
                    <h3 className="font-medium text-foreground">
                      {step.title}
                    </h3>
                  </div>
                  <p className="text-sm text-muted-foreground">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="w-full py-24 px-8 bg-background">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-12">
            {[
              { value: "10", label: "Shot Types" },
              { value: "33", label: "Landmarks" },
              { value: "20", label: "FPS" },
              { value: "100", label: "Max Score" },
            ].map((stat, idx) => (
              <div key={idx} className="space-y-2">
                <div className="text-4xl font-semibold text-foreground tabular-nums">
                  {stat.value}
                </div>
                <div className="text-sm text-muted-foreground">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full py-24 px-8 bg-muted/30">
        <div className="max-w-3xl mx-auto text-center space-y-6">
          <h2 className="text-3xl md:text-4xl font-semibold text-foreground">
            Ready to improve your game?
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Start analyzing your cricket technique today with AI-powered
            coaching and real-time feedback.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center pt-4">
            <Link
              to="/sports-coach"
              className="inline-flex items-center justify-center bg-primary text-primary-foreground font-medium h-11 px-8 rounded-md transition-colors hover:bg-primary/90"
            >
              Start Live Coaching
            </Link>
            <Link
              to="/dashboard"
              className="inline-flex items-center justify-center bg-background text-foreground font-medium h-11 px-8 rounded-md border border-border transition-colors hover:bg-accent"
            >
              View Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="w-full py-12 px-8 border-t border-border bg-background">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-sm text-muted-foreground">
              © 2025 SPORDO. AI-Powered Sports Assisstant.
            </div>
            <div className="text-sm text-muted-foreground">
              Built by Atharva, Ronit and Varun
            </div>
          </div>
        </div>
      </footer>
    </>
  );
};

// Utility function to shuffle an array
function shuffle(array) {
  let currentIndex = array.length,
    randomIndex;
  while (currentIndex !== 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex],
      array[currentIndex],
    ];
  }
  return array;
}

const squareData = [
  {
    id: 1,
    src: "grid/img1.avif",
  },
  {
    id: 2,
    src: "grid/img2.avif",
  },
  {
    id: 3,
    src: "grid/img3.avif",
  },
  {
    id: 4,
    src: "grid/img4.avif",
  },
  {
    id: 5,
    src: "grid/img5.avif",
  },
  {
    id: 6,
    src: "grid/img6.avif",
  },
  {
    id: 7,
    src: "grid/img7.avif",
  },
  {
    id: 8,
    src: "grid/img8.avif",
  },
  {
    id: 9,
    src: "grid/img9.avif",
  },
  {
    id: 10,
    src: "grid/img10.avif",
  },
  {
    id: 11,
    src: "grid/img11.avif",
  },
  {
    id: 12,
    src: "grid/img12.avif",
  },
  {
    id: 13,
    src: "grid/img13.avif",
  },
  {
    id: 14,
    src: "grid/img14.avif",
  },
  {
    id: 15,
    src: "grid/img15.avif",
  },
  {
    id: 16,
    src: "grid/img16.avif",
  },
];

const generateSquares = () => {
  return shuffle(squareData).map((sq) => (
    <motion.div
      key={sq.id}
      layout
      transition={{ duration: 1.5, type: "spring" }}
      className="w-full h-full"
      style={{
        backgroundImage: `url(${sq.src})`,
        backgroundSize: "cover",
      }}
    ></motion.div>
  ));
};

const ShuffleGrid = () => {
  const timeoutRef = useRef(null);
  const [squares, setSquares] = useState(generateSquares());

  useEffect(() => {
    shuffleSquares();

    return () => clearTimeout(timeoutRef.current);
  }, []);

  const shuffleSquares = () => {
    setSquares(generateSquares());

    timeoutRef.current = setTimeout(shuffleSquares, 3000);
  };

  return (
    <div className="grid grid-cols-4 grid-rows-4 h-[450px] gap-1">
      {squares.map((sq) => sq)}
    </div>
  );
};

export default Home;
