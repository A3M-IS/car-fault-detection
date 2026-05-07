import '../styles/globals.css'
import { Inter, Outfit } from 'next/font/google'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const outfit = Outfit({ subsets: ['latin'], variable: '--font-outfit' })

function MyApp({ Component, pageProps }) {
  return (
    <main className={`${inter.variable} ${outfit.variable}`}>
      <Component {...pageProps} />
    </main>
  )
}

export default MyApp
