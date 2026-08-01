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
      fontFamily: {
        sans: ['"Noto Sans SC"', 'PingFang SC', 'Microsoft YaHei', 'sans-serif'],
        display: ['"Noto Serif SC"', '"Noto Sans SC"', 'serif'],
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'soft-pulse': {
          '0%, 100%': { opacity: '0.45', transform: 'scale(1)' },
          '50%': { opacity: '0.7', transform: 'scale(1.03)' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.55s ease-out both',
        'fade-up-delay': 'fade-up 0.55s ease-out 0.12s both',
        'fade-up-delay-2': 'fade-up 0.55s ease-out 0.22s both',
        'soft-pulse': 'soft-pulse 6s ease-in-out infinite',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
