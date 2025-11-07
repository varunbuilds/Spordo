import { AppSidebar } from "../components/app-sidebar";
import { SiteHeader } from "../components/site-header";
import { SidebarInset, SidebarProvider } from "../components/ui/sidebar";
import { useState } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "../components/ui/card";
import { Button } from "../components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";
import { Badge } from "../components/ui/badge";

// Sample teams data - can be replaced with API data
const teams = [
  "Arsenal",
  "Aston Villa",
  "Bournemouth",
  "Brentford",
  "Brighton",
  "Chelsea",
  "Crystal Palace",
  "Everton",
  "Fulham",
  "Leeds United",
  "Leicester City",
  "Liverpool",
  "Man City",
  "Man United",
  "Newcastle",
  "Nottingham Forest",
  "Southampton",
  "Tottenham",
  "West Ham",
  "Wolves",
];

const LivePredictions = () => {
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!homeTeam || !awayTeam) {
      alert("Please select both teams");
      return;
    }

    if (homeTeam === awayTeam) {
      alert("Please select different teams");
      return;
    }

    setLoading(true);

    // Simulate API call - replace with actual API endpoint
    setTimeout(() => {
      setPrediction({
        matchup: `${homeTeam} vs ${awayTeam}`,
        quantModel: {
          homeWin: 60.81,
          draw: 20.43,
          awayWin: 18.75,
          confidence: 54.26,
        },
        fusedPrediction: {
          homeWin: 61.11,
          draw: 20.59,
          awayWin: 18.3,
          result: "H",
        },
        newsAnalysis: {
          homeTeam: {
            name: homeTeam,
            weight: 1.0,
            news: `${homeTeam} starting XI confirmed. Team News and Predicted Lineup. What channel is ${homeTeam} match on? Kick-off time and live stream details.`,
          },
          awayTeam: {
            name: awayTeam,
            weight: 0.97,
            news: `${awayTeam} team news, injury list, and predicted lineups. Latest updates on key players and match preparation.`,
          },
        },
      });
      setLoading(false);
    }, 1000);
  };

  const resetPrediction = () => {
    setHomeTeam("");
    setAwayTeam("");
    setPrediction(null);
  };

  return (
    <SidebarProvider
      style={{
        "--sidebar-width": "calc(var(--spacing) * 72)",
        "--header-height": "calc(var(--spacing) * 12)",
      }}
    >
      <AppSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader title="Live Predictions" />
        <div className="flex flex-1 flex-col p-4 md:p-6">
          {/* Header Card */}
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>EPL Hybrid Predictor</CardTitle>
              <CardDescription>
                Select two teams to get a quantitative (stats-based) and a final
                (fused with live news) prediction.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Home Team Select */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Home Team:</label>
                    <Select value={homeTeam} onValueChange={setHomeTeam}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select home team" />
                      </SelectTrigger>
                      <SelectContent>
                        {teams.map((team) => (
                          <SelectItem key={team} value={team}>
                            {team}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Away Team Select */}
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Away Team:</label>
                    <Select value={awayTeam} onValueChange={setAwayTeam}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select away team" />
                      </SelectTrigger>
                      <SelectContent>
                        {teams.map((team) => (
                          <SelectItem key={team} value={team}>
                            {team}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button onClick={handlePredict} disabled={loading}>
                    {loading ? "Predicting..." : "Predict"}
                  </Button>
                  {prediction && (
                    <Button variant="outline" onClick={resetPrediction}>
                      Reset
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Prediction Results */}
          {prediction && (
            <div className="space-y-6">
              {/* Match Title */}
              <div className="text-center">
                <h2 className="text-2xl font-bold">
                  Prediction: {prediction.matchup}
                </h2>
              </div>

              {/* Prediction Cards */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Quant-Only Prediction */}
                <Card>
                  <CardHeader>
                    <CardTitle>
                      Quant-Only Prediction ({prediction.quantModel.confidence}%
                      Model)
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Home Win:</span>
                      <Badge variant="default" className="text-base">
                        {prediction.quantModel.homeWin}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Draw:</span>
                      <Badge variant="secondary" className="text-base">
                        {prediction.quantModel.draw}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Away Win:</span>
                      <Badge variant="outline" className="text-base">
                        {prediction.quantModel.awayWin}%
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                {/* Final Fused Prediction */}
                <Card>
                  <CardHeader className="border-b">
                    <CardTitle className="flex items-center gap-2">
                      FINAL Fused Prediction (Quant + News)
                    </CardTitle>
                    <CardDescription>
                      Combined prediction using statistical model and live news
                      analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Home Win:</span>
                      <Badge variant="default" className="text-base px-3 py-1">
                        {prediction.fusedPrediction.homeWin}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Draw:</span>
                      <Badge
                        variant="secondary"
                        className="text-base px-3 py-1"
                      >
                        {prediction.fusedPrediction.draw}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Away Win:</span>
                      <Badge variant="outline" className="text-base px-3 py-1">
                        {prediction.fusedPrediction.awayWin}%
                      </Badge>
                    </div>
                    <div className="pt-4 border-t mt-4">
                      <div className="text-center space-y-3">
                        <p className="text-sm font-medium text-muted-foreground">
                          Most Likely Result
                        </p>
                        <Badge
                          variant="default"
                          className="text-2xl px-6 py-2 font-bold"
                        >
                          {prediction.fusedPrediction.result}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Contextual News Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle>Contextual News Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Home Team News */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-lg">
                          {prediction.newsAnalysis.homeTeam.name}
                        </h3>
                        <Badge variant="secondary">
                          Weight: {prediction.newsAnalysis.homeTeam.weight}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground italic leading-relaxed">
                        {prediction.newsAnalysis.homeTeam.news}
                      </p>
                    </div>

                    {/* Away Team News */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-lg">
                          {prediction.newsAnalysis.awayTeam.name}
                        </h3>
                        <Badge variant="secondary">
                          Weight: {prediction.newsAnalysis.awayTeam.weight}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground italic leading-relaxed">
                        {prediction.newsAnalysis.awayTeam.news}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Predict Another Match Button */}
              <div className="text-center">
                <Button variant="link" onClick={resetPrediction}>
                  Predict Another Match
                </Button>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!prediction && !loading && (
            <Card className="mt-6">
              <CardContent className="flex items-center justify-center py-12">
                <p className="text-muted-foreground text-center">
                  Select both teams and click "Predict" to see the match
                  prediction results
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
};

export default LivePredictions;
