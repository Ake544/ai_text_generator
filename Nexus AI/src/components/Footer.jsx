import React from "react";
import { NavLink } from "react-router-dom";

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-gray-200 py-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="glass p-8 card-shadow rounded-lg">
            <h3 className="text-lg font-bold mb-4">About Us</h3>
            <ul>
              <li className="mb-4">
                <NavLink
                  to="/about"
                  className={({ isActive }) =>
                    `text-gray-200 hover:text-white transition duration-300 ${isActive ? 'text-white' : ''}`
                  }
                >
                  Learn More
                </NavLink>
              </li>
              <li className="mb-4">
                <NavLink
                  to="/contact"
                  className={({ isActive }) =>
                    `text-gray-200 hover:text-white transition duration-300 ${isActive ? 'text-white' : ''}`
                  }
                >
                  Get in Touch
                </NavLink>
              </li>
            </ul>
          </div>
          <div className="glass p-8 card-shadow rounded-lg">
            <h3 className="text-lg font-bold mb-4">Resources</h3>
            <ul>
              <li className="mb-4">
                <NavLink
                  to="/docs"
                  className={({ isActive }) =>
                    `text-gray-200 hover:text-white transition duration-300 ${isActive ? 'text-white' : ''}`
                  }
                >
                  Documentation
                </NavLink>
              </li>
              <li className="mb-4">
                <NavLink
                  to="/faq"
                  className={({ isActive }) =>
                    `text-gray-200 hover:text-white transition duration-300 ${isActive ? 'text-white' : ''}`
                  }
                >
                  FAQ
                </NavLink>
              </li>
            </ul>
          </div>
          <div className="glass p-8 card-shadow rounded-lg">
            <h3 className="text-lg font-bold mb-4">Follow Us</h3>
            <ul>
              <li className="mb-4">
                <a
                  href="https://twitter.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-200 hover:text-white transition duration-300"
                >
                  Twitter
                </a>
              </li>
              <li className="mb-4">
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-200 hover:text-white transition duration-300"
                >
                  GitHub
                </a>
              </li>
            </ul>
          </div>
        </div>
        <p className="text-gray-400 text-center mt-8">
          &copy; {new Date().getFullYear()} Nexus AI. All rights reserved.
        </p>
      </div>
    </footer>
  );
};

export default Footer;