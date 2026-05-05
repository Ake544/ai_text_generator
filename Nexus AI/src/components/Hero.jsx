import React from "react";
import { useNavigate } from "react-router-dom";

const Hero = () => {
  const navigate = useNavigate();

  return (
    <section className="h-screen glass flex flex-col justify-center items-center dark:bg-gray-900 dark:text-gray-100">
      <h1 className="text-5xl font-bold mb-4 card-shadow">Nexus AI</h1>
      <p className="text-lg text-gray-400 mb-8 text-center sm:w-1/2 md:w-1/3 lg:w-1/4 dark:text-gray-500">
        Revolutionizing the future of artificial intelligence
      </p>
      <button
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-300 hover:scale-105 active:scale-95"
        onClick={() => navigate("/about")}
      >
        Learn More
      </button>
    </section>
  );
};

export default Hero;