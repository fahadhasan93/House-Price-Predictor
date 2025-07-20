"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Home, TrendingUp, BarChart3, Calculator } from "lucide-react"

interface HouseFeatures {
  bedrooms: number
  bathrooms: number
  sqft_living: number
  sqft_lot: number
  floors: number
  waterfront: number
  view: number
  condition: number
  grade: number
  age: number
  renovated: number
}

interface PredictionResult {
  predicted_price: number
  confidence_interval: [number, number]
  features_used: HouseFeatures
}

export default function HousePricePredictorApp() {
  const [features, setFeatures] = useState<HouseFeatures>({
    bedrooms: 3,
    bathrooms: 2,
    sqft_living: 2000,
    sqft_lot: 5000,
    floors: 1,
    waterfront: 0,
    view: 0,
    condition: 3,
    grade: 7,
    age: 20,
    renovated: 0,
  })

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (field: keyof HouseFeatures, value: string | number) => {
    setFeatures((prev) => ({
      ...prev,
      [field]: typeof value === "string" ? Number.parseFloat(value) || 0 : value,
    }))
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(features),
      })

      if (!response.ok) {
        throw new Error("Failed to get prediction")
      }

      const result = await response.json()
      setPrediction(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Home className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">House Price Predictor</h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Get accurate house price predictions using advanced machine learning models trained on real estate data
          </p>
        </div>

        <Tabs defaultValue="predict" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="predict" className="flex items-center gap-2">
              <Calculator className="h-4 w-4" />
              Price Prediction
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Model Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="predict" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Form */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Home className="h-5 w-5" />
                    House Features
                  </CardTitle>
                  <CardDescription>Enter the details of the house to get a price prediction</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="bedrooms">Bedrooms</Label>
                      <Input
                        id="bedrooms"
                        type="number"
                        min="1"
                        max="10"
                        value={features.bedrooms}
                        onChange={(e) => handleInputChange("bedrooms", e.target.value)}
                      />
                    </div>
                    <div>
                      <Label htmlFor="bathrooms">Bathrooms</Label>
                      <Input
                        id="bathrooms"
                        type="number"
                        min="1"
                        max="10"
                        step="0.5"
                        value={features.bathrooms}
                        onChange={(e) => handleInputChange("bathrooms", e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="sqft_living">Living Area (sq ft)</Label>
                      <Input
                        id="sqft_living"
                        type="number"
                        min="500"
                        value={features.sqft_living}
                        onChange={(e) => handleInputChange("sqft_living", e.target.value)}
                      />
                    </div>
                    <div>
                      <Label htmlFor="sqft_lot">Lot Size (sq ft)</Label>
                      <Input
                        id="sqft_lot"
                        type="number"
                        min="1000"
                        value={features.sqft_lot}
                        onChange={(e) => handleInputChange("sqft_lot", e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="floors">Floors</Label>
                      <Select
                        value={features.floors.toString()}
                        onValueChange={(value) => handleInputChange("floors", Number.parseFloat(value))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">1</SelectItem>
                          <SelectItem value="1.5">1.5</SelectItem>
                          <SelectItem value="2">2</SelectItem>
                          <SelectItem value="2.5">2.5</SelectItem>
                          <SelectItem value="3">3</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="age">Age (years)</Label>
                      <Input
                        id="age"
                        type="number"
                        min="0"
                        max="150"
                        value={features.age}
                        onChange={(e) => handleInputChange("age", e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="condition">Condition (1-5)</Label>
                      <Select
                        value={features.condition.toString()}
                        onValueChange={(value) => handleInputChange("condition", Number.parseInt(value))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">1 - Poor</SelectItem>
                          <SelectItem value="2">2 - Fair</SelectItem>
                          <SelectItem value="3">3 - Average</SelectItem>
                          <SelectItem value="4">4 - Good</SelectItem>
                          <SelectItem value="5">5 - Excellent</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="grade">Grade (3-12)</Label>
                      <Input
                        id="grade"
                        type="number"
                        min="3"
                        max="12"
                        value={features.grade}
                        onChange={(e) => handleInputChange("grade", e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="waterfront">Waterfront</Label>
                      <Select
                        value={features.waterfront.toString()}
                        onValueChange={(value) => handleInputChange("waterfront", Number.parseInt(value))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">No</SelectItem>
                          <SelectItem value="1">Yes</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="view">View (0-4)</Label>
                      <Select
                        value={features.view.toString()}
                        onValueChange={(value) => handleInputChange("view", Number.parseInt(value))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">0 - None</SelectItem>
                          <SelectItem value="1">1 - Fair</SelectItem>
                          <SelectItem value="2">2 - Average</SelectItem>
                          <SelectItem value="3">3 - Good</SelectItem>
                          <SelectItem value="4">4 - Excellent</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="renovated">Renovated</Label>
                      <Select
                        value={features.renovated.toString()}
                        onValueChange={(value) => handleInputChange("renovated", Number.parseInt(value))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">No</SelectItem>
                          <SelectItem value="1">Yes</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <Button onClick={handlePredict} disabled={loading} className="w-full" size="lg">
                    {loading ? "Predicting..." : "Predict House Price"}
                  </Button>
                </CardContent>
              </Card>

              {/* Results */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Price Prediction
                  </CardTitle>
                  <CardDescription>AI-powered price estimation based on your inputs</CardDescription>
                </CardHeader>
                <CardContent>
                  {error && (
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-red-600">{error}</p>
                    </div>
                  )}

                  {prediction && (
                    <div className="space-y-4">
                      <div className="text-center p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border">
                        <div className="text-3xl font-bold text-gray-900 mb-2">
                          {formatPrice(prediction.predicted_price)}
                        </div>
                        <p className="text-gray-600">Estimated Market Value</p>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-gray-50 rounded-lg">
                          <div className="text-sm text-gray-600">Price Range</div>
                          <div className="font-semibold">
                            {formatPrice(prediction.confidence_interval[0])} -{" "}
                            {formatPrice(prediction.confidence_interval[1])}
                          </div>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg">
                          <div className="text-sm text-gray-600">Price per Sq Ft</div>
                          <div className="font-semibold">
                            {formatPrice(prediction.predicted_price / features.sqft_living)}
                          </div>
                        </div>
                      </div>

                      <Separator />

                      <div>
                        <h4 className="font-semibold mb-2">Key Features</h4>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="secondary">{features.bedrooms} bed</Badge>
                          <Badge variant="secondary">{features.bathrooms} bath</Badge>
                          <Badge variant="secondary">{features.sqft_living.toLocaleString()} sq ft</Badge>
                          <Badge variant="secondary">{features.age} years old</Badge>
                          {features.waterfront === 1 && <Badge variant="default">Waterfront</Badge>}
                          {features.renovated === 1 && <Badge variant="default">Renovated</Badge>}
                        </div>
                      </div>
                    </div>
                  )}

                  {!prediction && !error && (
                    <div className="text-center py-8 text-gray-500">
                      <Home className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>Enter house details and click "Predict House Price" to get started</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="analysis">
            <Card>
              <CardHeader>
                <CardTitle>Model Performance & Analysis</CardTitle>
                <CardDescription>Insights into the machine learning model used for predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">Random Forest</div>
                    <div className="text-sm text-gray-600">Model Type</div>
                  </div>
                  <div className="p-4 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">85%+</div>
                    <div className="text-sm text-gray-600">Accuracy (R² Score)</div>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">1000+</div>
                    <div className="text-sm text-gray-600">Training Samples</div>
                  </div>
                </div>

                <Separator className="my-6" />

                <div>
                  <h4 className="font-semibold mb-4">Key Features Importance</h4>
                  <div className="space-y-3">
                    {[
                      { feature: "Living Area (sq ft)", importance: 0.35 },
                      { feature: "Grade", importance: 0.18 },
                      { feature: "Bathrooms", importance: 0.12 },
                      { feature: "Age", importance: 0.1 },
                      { feature: "Bedrooms", importance: 0.08 },
                      { feature: "Condition", importance: 0.07 },
                      { feature: "View", importance: 0.05 },
                      { feature: "Waterfront", importance: 0.05 },
                    ].map((item, index) => (
                      <div key={index} className="flex items-center gap-3">
                        <div className="w-32 text-sm">{item.feature}</div>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${item.importance * 100}%` }}
                          />
                        </div>
                        <div className="text-sm font-medium">{(item.importance * 100).toFixed(0)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
