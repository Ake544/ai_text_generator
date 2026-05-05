import React from "react";
import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";

import Hero from "./Hero";
import Footer from "./Footer";

function App() {
  return (
    <div className="min-h-screen bg-black text-white glass">
      <BrowserRouter basename={import.meta.env.BASE_URL}>
        <Routes>
          <Route path="/" element={<Hero />} />
        </Routes>
      </BrowserRouter>
      <Footer className="card-shadow" />
    </div>
  );
}

export default App;