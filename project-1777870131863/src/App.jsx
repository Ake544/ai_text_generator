import React from "react";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { FiEdit, FiCheck } from "lucide-react";

function Home() {
  const navigate = useNavigate();
  const [text, setText] = React.useState("");
  const [loading, setLoading] = React.useState(false);

  const handleNavigate = () => {
    navigate("/editor", { state: { text } });
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="h-screen w-screen bg-gray-900 flex flex-col justify-center items-center"
    >
      <div className="glass p-10 rounded-lg card-shadow w-1/2 md:w-1/3 lg:w-1/4">
        <h1 className="text-3xl text-white font-bold mb-4">Text Editor</h1>
        <textarea
          className="w-full h-48 p-4 rounded-lg bg-gray-700 text-white resize-none"
          placeholder="Write your text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          className="mt-4 py-2 px-4 rounded-lg bg-blue-500 text-white hover:bg-blue-700"
          onClick={handleNavigate}
          disabled={loading}
        >
          {loading ? (
            <svg
              className="animate-spin h-5 w-5 mr-3 border-4 border-gray-200 rounded-full border-t-blue-600"
              viewBox="0 0 24 24"
            />
          ) : (
            <FiEdit size={20} className="mr-2" />
          )}
          {loading ? "Loading..." : "Edit Text"}
        </button>
      </div>
    </motion.div>
  );
}

function Editor() {
  const navigate = useNavigate();
  const location = useLocation();
  const [text, setText] = React.useState(location.state?.text || "");
  const [loading, setLoading] = React.useState(false);

  const handleBack = () => {
    navigate(-1);
  };

  const handleSave = () => {
    setLoading(true);
    const savedText = text;
    localStorage.setItem("savedText", savedText);
    console.log("Text saved:", savedText);
    setLoading(false);
    navigate(-1);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="h-screen w-screen bg-gray-900 flex flex-col justify-center items-center"
    >
      <div className="glass p-10 rounded-lg card-shadow w-1/2 md:w-1/3 lg:w-1/4">
        <h1 className="text-3xl text-white font-bold mb-4">Text Editor</h1>
        <textarea
          className="w-full h-48 p-4 rounded-lg bg-gray-700 text-white resize-none"
          defaultValue={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="flex justify-between mt-4">
          <button
            className="py-2 px-4 rounded-lg bg-gray-500 text-white hover:bg-gray-700"
            onClick={handleBack}
          >
            Back
          </button>
          <button
            className="py-2 px-4 rounded-lg bg-blue-500 text-white hover:bg-blue-700"
            onClick={handleSave}
            disabled={loading}
          >
            {loading ? (
              <svg
                className="animate-spin h-5 w-5 mr-3 border-4 border-gray-200 rounded-full border-t-blue-600"
                viewBox="0 0 24 24"
              />
            ) : (
              <FiCheck size={20} className="mr-2" />
            )}
            {loading ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </motion.div>
  );
}

function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/editor" element={<Editor />} />
      </Routes>
    </BrowserRouter>
  );
}

export { Home, Editor };
export default App;