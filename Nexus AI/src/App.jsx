import React from "react";
import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";

import Hero from "./components/Hero";
import Footer from "./components/Footer";

function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
        <div className="flex-1 container mx-auto p-4 pt-6 md:p-6 lg:p-12 lg:pt-20 glass card-shadow">
          <Routes>
            <Route path="/" element={<Hero />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;