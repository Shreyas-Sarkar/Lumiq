import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "lumiq-purple": "#7C3AED",
        "lumiq-dark": "#0F0F0F",
        "lumiq-gray": "#1E1E1E",
      },
    },
  },
  plugins: [],
};
export default config;
