/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      colors: {
        ink: {
          900: "#0a0a0b",
          800: "#111114",
          700: "#1a1a1f",
          600: "#26262d",
          500: "#3a3a44",
          400: "#6e6e7a",
          300: "#a8a8b3",
          200: "#d4d4dc",
          100: "#f4f4f7",
        },
        accent: {
          load: "#5fb3ff",
          compute: "#ffb86b",
          store: "#a78bfa",
        },
      },
    },
  },
  plugins: [],
};
