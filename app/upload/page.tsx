'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { NicheSelector } from '../components/NicheSelector';
import { UsageTracker, useUsageTracking } from '../components/UsageTracker';
import { useFormAnalytics, useUploadAnalytics } from '@/lib/hooks/useAnalytics';

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [videoTitle, setVideoTitle] = useState('');
  const [selectedNiche, setSelectedNiche] = useState('general');
  const router = useRouter();
  
  const formAnalytics = useFormAnalytics('upload_form');
  const uploadAnalytics = useUploadAnalytics();
  const { usage, refreshUsage, checkCanAnalyze } = useUsageTracking();

  useEffect(() => {
    // Track form start
    try {
      formAnalytics.trackFormStart();
    } catch (error) {
      console.warn('Analytics error:', error);
    }
  }, []); // Remove formAnalytics dependency to prevent infinite re-renders

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).slice(0, 3); // Limit to 3 files
      setFiles(newFiles);
      
      // Track file selection
      try {
        const totalSize = newFiles.reduce((sum, file) => sum + file.size, 0);
        uploadAnalytics.trackFileSelection(newFiles.length, totalSize);
        
        // Track niche selection
        formAnalytics.trackFormFieldInteraction('file_upload', 'files_selected');
      } catch (error) {
        console.warn('Analytics error:', error);
      }
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
    try {
      formAnalytics.trackFormFieldInteraction('file_upload', 'file_removed');
    } catch (error) {
      console.warn('Analytics error:', error);
    }
  };

  const handleNicheChange = (niche: string) => {
    setSelectedNiche(niche);
    try {
      formAnalytics.trackFormFieldInteraction('niche_selector', niche);
    } catch (error) {
      console.warn('Analytics error:', error);
    }
  };

  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setVideoTitle(e.target.value);
    try {
      formAnalytics.trackFormFieldInteraction('video_title', 'title_entered');
    } catch (error) {
      console.warn('Analytics error:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;

    // Check usage limits before uploading
    if (!checkCanAnalyze()) {
      alert('You have reached your monthly analysis limit. Please upgrade your plan or wait until next month.');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    files.forEach((file, index) => {
      formData.append(`file${index}`, file);
    });

    try {
      // Get session ID for usage tracking
      const sessionId = localStorage.getItem('session-id') || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('session-id', sessionId);
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        headers: {
          'X-Session-Id': sessionId,
        },
      });

      const data = await response.json();
      
      if (response.ok) {
        // Track successful upload
        try {
          const fileSizes = files.map(file => file.size);
          formAnalytics.trackFormSubmit(true);
        } catch (error) {
          console.warn('Analytics error:', error);
        }
        
        // Store upload data in session storage for analysis
        if (videoTitle) {
          sessionStorage.setItem('videoTitle', videoTitle);
        }
        if (selectedNiche) {
          sessionStorage.setItem('selectedNiche', selectedNiche);
        }
        if (data.thumbnails) {
          sessionStorage.setItem('thumbnails', JSON.stringify(data.thumbnails));
        }

        // Store image URLs for display in results
        const imageUrls: string[] = [];
        files.forEach((file) => {
          const url = URL.createObjectURL(file);
          imageUrls.push(url);
        });
        sessionStorage.setItem('imageUrls', JSON.stringify(imageUrls));
        
        // Refresh usage after successful upload
        await refreshUsage();
        
        // Redirect to results page
        router.push(`/results?id=${data.sessionId}`);
      } else {
        // Track upload error
        try {
          formAnalytics.trackFormSubmit(false, data.error);
          uploadAnalytics.trackUploadError(data.error);
        } catch (analyticsError) {
          console.warn('Analytics error:', analyticsError);
        }
        alert('Upload failed: ' + data.error);
      }
    } catch (error) {
      // Track upload error
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      try {
        formAnalytics.trackFormSubmit(false, errorMessage);
        uploadAnalytics.trackUploadError(errorMessage);
      } catch (analyticsError) {
        console.warn('Analytics error:', analyticsError);
      }
      alert('Upload failed: ' + error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm mb-6 inline-block transition-colors duration-200">
            ← Back to Home
          </Link>
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-[#00C8C8] via-[#66B2FF] to-[#00C8C8] bg-clip-text text-transparent">
            Thumbscore.io
          </h1>
          <p className="text-xl text-gray-300 mb-6 font-medium">Upload 3 YouTube Thumbnails</p>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto leading-relaxed">
            Upload 3 different thumbnail options to see which will optimize your YouTube click-through rate
          </p>
        </div>
        
        {/* Usage Tracker */}
        <div className="mb-8">
          <UsageTracker />
        </div>
        
        {/* Main Form Container */}
        <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 shadow-2xl">
          {/* Enhanced Niche Selector */}
          <div className="mb-8">
            <NicheSelector 
              value={selectedNiche}
              onChange={handleNicheChange}
            />
          </div>
          
          {/* Video Title Input */}
          <div className="mb-8">
            <label htmlFor="video-title" className="block text-lg font-semibold text-white mb-3">
              Video Title (Optional - for better analysis)
            </label>
            <input
              type="text"
              id="video-title"
              value={videoTitle}
              onChange={handleTitleChange}
              placeholder="Enter your YouTube video title..."
              className="w-full px-6 py-4 bg-gray-700/50 border border-gray-600/50 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200 text-lg"
            />
            <p className="text-sm text-gray-400 mt-2 flex items-center gap-2">
              <span className="w-4 h-4 bg-blue-500/20 rounded-full flex items-center justify-center text-xs">i</span>
              Providing the title helps AI analyze title-thumbnail semantic matching
            </p>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Enhanced File Upload Area */}
            <div className="relative">
              <div className="border-2 border-dashed border-gray-500/50 rounded-2xl p-12 text-center hover:border-blue-500/50 transition-all duration-300 bg-gray-700/20">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer block"
                >
                  <div className="mb-4">
                    <div className="w-16 h-16 mx-auto bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mb-4">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      {files.length === 0 ? 'Choose up to 3 thumbnail files' : `${files.length}/3 thumbnails selected`}
                    </h3>
                    <p className="text-gray-400">
                      {files.length === 0 ? 'Click to browse or drag and drop your images here' : 'Click to change files'}
                    </p>
                  </div>
                </label>
              </div>
            </div>

            {/* Enhanced File Preview Grid */}
            {files.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {files.map((file, index) => (
                  <div key={index} className="relative group bg-gray-700/30 rounded-xl p-4 border border-gray-600/30 hover:border-blue-500/50 transition-all duration-200">
                    <div className="w-full h-40 bg-gray-600/30 rounded-lg mb-3 overflow-hidden">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={URL.createObjectURL(file)}
                        alt={`Thumbnail ${index + 1}`}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <p className="text-white text-sm font-medium truncate flex-1">{file.name}</p>
                      <button
                        type="button"
                        onClick={() => removeFile(index)}
                        className="ml-2 bg-red-600/80 text-white rounded-full w-7 h-7 text-sm hover:bg-red-600 transition-colors duration-200 flex items-center justify-center"
                      >
                        ×
                      </button>
                    </div>
                    <div className="mt-2 text-xs text-gray-400">
                      {(file.size / 1024 / 1024).toFixed(1)} MB
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Enhanced Analyze Button */}
            <button
              type="submit"
              disabled={files.length === 0 || uploading}
              className="w-full px-8 py-4 bg-gradient-to-r from-purple-600 via-blue-600 to-teal-600 text-white rounded-xl font-semibold text-lg hover:from-purple-500 hover:via-blue-500 hover:to-teal-500 disabled:from-gray-600 disabled:via-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] disabled:hover:scale-100 shadow-lg hover:shadow-xl"
            >
              {uploading ? (
                <div className="flex items-center justify-center gap-3">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  Analyzing Thumbnails...
                </div>
              ) : (
                `Analyze ${files.length} Thumbnail${files.length !== 1 ? 's' : ''}`
              )}
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}

