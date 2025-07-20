import { type NextRequest, NextResponse } from "next/server"

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

// Simple prediction function (in production, you'd load the actual trained model)
function predictHousePrice(features: HouseFeatures): number {
  // This is a simplified version of the model for demonstration
  // In production, you would load the actual trained model using joblib or similar

  const basePrice =
    features.bedrooms * 45000 +
    features.bathrooms * 35000 +
    features.sqft_living * 140 +
    features.sqft_lot * 4 +
    features.floors * 18000 +
    features.waterfront * 180000 +
    features.view * 22000 +
    features.condition * 12000 +
    features.grade * 38000 +
    features.age * -800 +
    features.renovated * 45000 +
    100000 // Base price

  // Add some realistic variation
  const variation = Math.random() * 0.1 - 0.05 // ±5% variation
  return Math.max(basePrice * (1 + variation), 50000)
}

export async function POST(request: NextRequest) {
  try {
    const features: HouseFeatures = await request.json()

    // Validate input
    const requiredFields = [
      "bedrooms",
      "bathrooms",
      "sqft_living",
      "sqft_lot",
      "floors",
      "waterfront",
      "view",
      "condition",
      "grade",
      "age",
      "renovated",
    ]

    for (const field of requiredFields) {
      if (!(field in features) || typeof features[field as keyof HouseFeatures] !== "number") {
        return NextResponse.json({ error: `Missing or invalid field: ${field}` }, { status: 400 })
      }
    }

    // Make prediction
    const predictedPrice = predictHousePrice(features)

    // Calculate confidence interval (±15% for demonstration)
    const confidenceRange = predictedPrice * 0.15
    const confidenceInterval: [number, number] = [predictedPrice - confidenceRange, predictedPrice + confidenceRange]

    const result = {
      predicted_price: Math.round(predictedPrice),
      confidence_interval: [Math.round(confidenceInterval[0]), Math.round(confidenceInterval[1])] as [number, number],
      features_used: features,
    }

    return NextResponse.json(result)
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json({ error: "Failed to make prediction" }, { status: 500 })
  }
}
