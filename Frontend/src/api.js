const API_BASE = process.env.NODE_ENV === 'production' 
  ? "https://your-backend-app-name.herokuapp.com" 
  : "http://localhost:8000";

export async function getWeatherPrediction(city) {
  try {
    const response = await fetch(`${API_BASE}/weather/predict/${city}`);
    if (!response.ok) throw new Error("API error");
    return await response.json();
  } catch (error) {
    console.error("Error fetching prediction:", error);
    return { error: error.message };
  }
}

