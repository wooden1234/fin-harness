/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          navy: '#1E3A5F',
          gold: '#C9A227',
          light: '#2D5A87',
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
