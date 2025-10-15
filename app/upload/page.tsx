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
    formAnalytics.trackFormStart();
  }, [formAnalytics]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).slice(0, 3); // Limit to 3 files
      setFiles(newFiles);
      
      // Track file selection
      const totalSize = newFiles.reduce((sum, file) => sum + file.size, 0);
      uploadAnalytics.trackFileSelection(newFiles.length, totalSize);
      
      // Track niche selection
      formAnalytics.trackFormFieldInteraction('file_upload', 'files_selected');
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
    formAnalytics.trackFormFieldInteraction('file_upload', 'file_removed');
  };

  const handleNicheChange = (niche: string) => {
    setSelectedNiche(niche);
    formAnalytics.trackFormFieldInteraction('niche_selector', niche);
  };

  const handleTitleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setVideoTitle(e.target.value);
    formAnalytics.trackFormFieldInteraction('video_title', 'title_entered');
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
        const fileSizes = files.map(file => file.size);
        formAnalytics.trackFormSubmit(true);
        
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
        formAnalytics.trackFormSubmit(false, data.error);
        uploadAnalytics.trackUploadError(data.error);
        alert('Upload failed: ' + data.error);
      }
    } catch (error) {
      // Track upload error
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      formAnalytics.trackFormSubmit(false, errorMessage);
      uploadAnalytics.trackUploadError(errorMessage);
      alert('Upload failed: ' + error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex flex-col items-center justify-center p-24">
      <div className="w-full max-w-4xl">
        <div className="text-center mb-8">
          <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm mb-4 inline-block">
            ← Back to Home
          </Link>
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-[#6a5af9] via-[#1de9b6] to-[#6a5af9] bg-clip-text text-transparent">
            Thumbscore.io
          </h1>
          <p className="text-lg text-gray-300 mb-4">Upload 3 Thumbnails</p>
        </div>
        <p className="text-center text-gray-400 mb-8">
          Upload 3 different thumbnail options to see which will get more clicks on YouTube
        </p>
        
        {/* Usage Tracker */}
        <div className="mb-6">
          <UsageTracker />
        </div>
        
        {/* Enhanced Niche Selector */}
        <NicheSelector 
          value={selectedNiche}
          onChange={handleNicheChange}
        />
        
        {/* Video Title Input */}
        <div className="mb-6">
          <label htmlFor="video-title" className="block text-sm font-medium text-gray-300 mb-2">
            Video Title (Optional - for better analysis)
          </label>
          <input
            type="text"
            id="video-title"
            value={videoTitle}
            onChange={handleTitleChange}
            placeholder="Enter your YouTube video title..."
            className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Providing the title helps AI analyze title-thumbnail semantic matching
          </p>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="border-2 border-dashed border-white rounded-lg p-8 text-center">
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
              className="cursor-pointer text-blue-400 hover:text-blue-300 block"
            >
              {files.length === 0 ? 'Choose up to 3 thumbnail files' : `${files.length}/3 thumbnails selected`}
            </label>
          </div>

              {files.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {files.map((file, index) => (
                    <div key={index} className="relative bg-gray-800 rounded-lg p-4">
                      <div className="w-full h-32 bg-gray-700 rounded mb-2 overflow-hidden">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={URL.createObjectURL(file)}
                          alt={`Thumbnail ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <p className="text-white text-sm truncate">{file.name}</p>
                      <button
                        type="button"
                        onClick={() => removeFile(index)}
                        className="absolute top-2 right-2 bg-red-600 text-white rounded-full w-6 h-6 text-xs hover:bg-red-700"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}

          <button
            type="submit"
            disabled={files.length === 0 || uploading}
            className="w-full px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-800 transition-colors"
          >
            {uploading ? 'Analyzing Thumbnails...' : `Analyze ${files.length} Thumbnail${files.length !== 1 ? 's' : ''}`}
          </button>
        </form>
      </div>
    </main>
  );
}

