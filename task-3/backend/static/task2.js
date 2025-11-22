import { demoUsers, demoRestaurants } from "./data.js";

const personaCardsEl = document.getElementById("personaCards");
const personaNoteEl = document.getElementById("personaNote");
const restaurantSelectEl = document.getElementById("restaurantSelect");
const reviewTextEl = document.getElementById("reviewText");
const helpfulCountEl = document.getElementById("helpfulCount");
const seasonSelectEl = document.getElementById("seasonSelect");
const dayTypeSelectEl = document.getElementById("dayTypeSelect");
const weatherSelectEl = document.getElementById("weatherSelect");
const ratingStatusEl = document.getElementById("ratingStatus");
const ratingResultEl = document.getElementById("ratingResult");
const ratingForm = document.getElementById("ratingForm");

let selectedUser = null;
let selectedRestaurant = demoRestaurants[0];

function renderPersonas() {
  personaCardsEl.innerHTML = "";
  demoUsers.forEach((user) => {
    const card = document.createElement("div");
    card.className = "user-card";
    card.innerHTML = `
      <h4>${user.name}</h4>
      <p><strong>Location:</strong> ${user.home_location}</p>
      <p><strong>Favorites:</strong> ${user.favorite_cuisines.join(", ")}</p>
      <p><strong>Price:</strong> ${user.preferred_price_range}</p>
      <p><strong>Dietary:</strong> ${user.dietary_restrictions}</p>
    `;
    card.addEventListener("click", () => {
      [...personaCardsEl.children].forEach((child) => child.classList.remove("selected"));
      card.classList.add("selected");
      selectedUser = { ...user };
      personaNoteEl.textContent = `Persona selected: ${user.name}.`;
    });
    personaCardsEl.appendChild(card);
  });
}

function renderRestaurants() {
  restaurantSelectEl.innerHTML = "";
  demoRestaurants.forEach((rest, idx) => {
    const option = document.createElement("option");
    option.value = rest.id;
    option.textContent = `${rest.name} — ${rest.cuisine} (${rest.location})`;
    if (idx === 0) option.selected = true;
    restaurantSelectEl.appendChild(option);
  });
  restaurantSelectEl.addEventListener("change", (ev) => {
    const id = Number(ev.target.value);
    selectedRestaurant = demoRestaurants.find((r) => r.id === id) || demoRestaurants[0];
  });
}

function buildRestaurantPayload(rest) {
  return {
    location: rest.location,
    cuisine: rest.cuisine,
    price_bucket: rest.price_bucket,
    description: rest.description,
    amenities: rest.amenities,
    attributes: rest.attributes,
    avg_price: rest.avg_price,
    popularity_score: rest.popularity_score,
    trend_features: rest.trend_features,
  };
}

function buildUserPayload(rest) {
  const base = selectedUser || {
    age: 34,
    home_location: rest.location,
    dining_frequency: "Monthly",
    favorite_cuisines: [rest.cuisine],
    preferred_price_range: rest.price_bucket,
    dietary_restrictions: "none",
    avg_rating_given: 3.6,
    total_reviews_written: 42,
  };
  const cuisineMatch = base.favorite_cuisines.includes(rest.cuisine) ? 1 : 0;
  return {
    age: base.age,
    home_location: base.home_location,
    preferred_price_range: base.preferred_price_range,
    dietary_restrictions: base.dietary_restrictions,
    dining_frequency: base.dining_frequency,
    avg_rating_given: base.avg_rating_given,
    total_reviews_written: base.total_reviews_written,
    is_local_resident: base.home_location === rest.location,
    user_cuisine_match: cuisineMatch,
    dietary_conflict: 0,
  };
}

function buildReviewContext() {
  const now = new Date();
  return {
    helpful_count: Number(helpfulCountEl.value) || 0,
    season: seasonSelectEl.value,
    day_type: dayTypeSelectEl.value,
    weather_impact_category: weatherSelectEl.value,
    review_month: now.getMonth() + 1,
    review_day_of_week: now.getDay(),
    is_holiday: false,
    booking_lead_time_days: 5,
  };
}

function renderRatingResult(data) {
  const payload = data.data;
  ratingResultEl.innerHTML = `
    <h4>Results</h4>
    <p>Continuous rating: <strong>${payload.rating_prediction.toFixed(2)}</strong></p>
    <p>Rounded star: <strong>${payload.rounded_rating.toFixed(0)} / 5</strong></p>
    ${
      payload.confidence_interval
        ? `<p>Confidence interval: ${payload.confidence_interval.map((v) => v.toFixed(2)).join(" – ")}</p>`
        : ""
    }
    <p><strong>Model:</strong> ${payload.model_version} (${payload.inference_mode})</p>
  `;
}

async function handleSubmit(ev) {
  ev.preventDefault();
  const reviewText = reviewTextEl.value.trim();
  if (!reviewText) return;

  const restaurant =
    demoRestaurants.find((r) => r.id === Number(restaurantSelectEl.value)) || selectedRestaurant;
  selectedRestaurant = restaurant;

  const payload = {
    restaurant: buildRestaurantPayload(restaurant),
    user: buildUserPayload(restaurant),
    review_context: buildReviewContext(),
    review_text: reviewText,
  };

  ratingStatusEl.textContent = "Scoring...";
  ratingStatusEl.className = "status";

  try {
    const response = await fetch("/v1/ratings/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`API error ${response.status}`);
    const data = await response.json();
    renderRatingResult(data);
    ratingStatusEl.textContent = `Trace ID: ${data.trace_id} · Latency: ${data.latency_ms}ms`;
    ratingStatusEl.className = "status success";
  } catch (err) {
    ratingStatusEl.textContent = `Prediction failed: ${err.message}`;
    ratingStatusEl.className = "status error";
  }
}

renderPersonas();
renderRestaurants();
ratingForm.addEventListener("submit", handleSubmit);

