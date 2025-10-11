export default function Home() {
  return (
    <main className="min-h-screen bg-black flex flex-col items-center justify-center p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm">
        <h1 className="text-5xl font-bold mb-8 text-center text-white">
          Thumbnail Lab
        </h1>
        <p className="text-center mb-4 text-gray-300 text-lg">
          Upload 3 thumbnail options and discover which one will get more clicks on YouTube
        </p>
        <p className="text-center mb-8 text-gray-400">
          AI-powered analysis to maximize your video&apos;s click-through rate
        </p>
        <div className="flex justify-center gap-4">
          <a
            href="/upload"
            className="px-8 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg font-semibold"
          >
            Test Your Thumbnails
          </a>
          <a
            href="/faq"
            className="px-8 py-4 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors text-lg font-semibold"
          >
            FAQ
          </a>
        </div>
        
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="text-3xl mb-4">📸</div>
            <h3 className="text-xl font-semibold mb-2 text-white">Upload 3 Options</h3>
            <p className="text-gray-400">Upload up to 3 different thumbnail variations to compare</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="text-3xl mb-4">🤖</div>
            <h3 className="text-xl font-semibold mb-2 text-white">AI Analysis</h3>
            <p className="text-gray-400">Our AI analyzes color, composition, and click potential</p>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="text-3xl mb-4">📈</div>
            <h3 className="text-xl font-semibold mb-2 text-white">Get Results</h3>
            <p className="text-gray-400">See which thumbnail is predicted to perform best</p>
          </div>
        </div>
      </div>
    </main>
  );
}
