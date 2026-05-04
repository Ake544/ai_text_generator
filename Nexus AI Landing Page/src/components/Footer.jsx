import React from "react";
import { motion } from "framer-motion";
import { Twitter, Linkedin, Github } from "lucide-react";
import clsx from "clsx";
import { twMerge as tailwindMerge } from "tailwind-merge";

// Define social media links and their respective icons
const socialLinks = [
  { name: "Twitter", icon: Twitter, url: "https://twitter.com/NexusAI" },
  { name: "LinkedIn", icon: Linkedin, url: "https://linkedin.com/company/NexusAI" },
  { name: "GitHub", icon: Github, url: "https://github.com/NexusAI" },
];

// Framer Motion variants for a subtle entrance animation for the footer itself
const footerVariants = {
  hidden: { opacity: 0, y: 50 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: "easeOut",
      when: "beforeChildren", // Animate children after the parent footer
      staggerChildren: 0.1, // Stagger children animations
    },
  },
};

// Framer Motion variants for child elements (copyright text, social icons)
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: "easeOut",
    },
  },
};

const Footer = () => {
  return (
    <motion.footer
      className={clsx(
        "w-full bg-gradient-to-r from-gray-900 to-gray-950 text-gray-200 py-8 px-4 sm:px-6 lg:px-8",
        "glass card-shadow"
      )}
      initial="hidden"
      animate="visible"
      variants={footerVariants}
    >
      <div className={tailwindMerge(
        {
          "max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between":
            true,
          "px-4 sm:px-6 lg:px-8 py-8": true,
          "bg-white/10 backdrop-filter backdrop-blur-md border border-gray-700/10 rounded-md glass":
            true,
        }
      )}
      >
        {/* Copyright Information */}
        <motion.p
          className="text-sm font-light mb-4 md:mb-0 text-gray-400 dark:text-gray-600"
          variants={itemVariants}
        >
          &copy; {new Date().getFullYear()} Nexus AI. All rights reserved.
        </motion.p>

        {/* Social Media Links */}
        <div className="flex space-x-6">
          {socialLinks.map((link) => (
            <motion.a
              key={link.name}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className={clsx(
                "text-gray-400 hover:text-blue-400 transition-colors duration-300 transform",
                "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 rounded-md"
              )}
              variants={itemVariants}
              whileHover={{ scale: 1.15, y: -5, transition: { type: "spring", stiffness: 400, damping: 10 } }}
              aria-label={`Link to Nexus AI's ${link.name}`}
            >
              <link.icon size={24} strokeWidth={1.5} />
            </motion.a>
          ))}
        </div>
      </div>
    </motion.footer>
  );
};

export default Footer;