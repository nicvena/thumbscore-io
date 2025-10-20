'use client';

import { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { NicheSelector } from '../components/NicheSelector';
import { UsageTracker, useUsageTracking } from '../components/UsageTracker';
import { useFormAnalytics, useUploadAnalytics } from '@/lib/hooks/useAnalytics';
import { useFreeCredits } from '@/lib/hooks/useFreeCredits';
import { useAuth } from '@/lib/hooks/useAuth';
import { UpgradeModal } from '../components/UpgradeModal';

export default function UploadPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [videoTitle, setVideoTitle] = useState('');
  const [selectedNiche, setSelectedNiche] = useState('general');
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const router = useRouter();
  
  // Temporarily disable analytics for performance testing
  const formAnalytics = { trackFormStart: () => {}, trackFormFieldInteraction: () => {}, trackFormSubmit: () => {} };
  const uploadAnalytics = { trackFileSelection: () => {}, trackUploadError: () => {} };
  const { usage, refreshUsage, checkCanAnalyze } = useUsageTracking();
  const { isUsedUp, increment } = useFreeCredits();
  const { user, hasUnlimitedAccess, getPlanDisplayName, signOut } = useAuth();

  // Compress image to reduce storage size
  const compressImage = async (file: File, maxWidth: number = 300, quality: number = 0.7): Promise<string> => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      
      img.onload = () => {
        // Calculate new dimensions
        const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;
        
        // Draw and compress
        ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
        const compressedDataUrl = canvas.toDataURL('image/jpeg', quality);
        resolve(compressedDataUrl);
      };
      
      img.onerror = () => {
        // Fallback to original file as data URL
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.readAsDataURL(file);
      };
      
      img.src = URL.createObjectURL(file);
    });
  };

  // Lazy image preview component that doesn't block initial file selection
  const LazyImagePreview = ({ file, index }: { file: File; index: number }) => {
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
      // Delay image preview creation to not block file selection
      const timer = setTimeout(() => {
        try {
          const url = URL.createObjectURL(file);
          setImageUrl(url);
          setIsLoading(false);
        } catch (error) {
          console.warn('Failed to create image preview:', error);
          setIsLoading(false);
        }
      }, 100 * index); // Stagger loading to prevent blocking

      return () => {
        clearTimeout(timer);
        // Cleanup URL when component unmounts
        if (imageUrl) {
          URL.revokeObjectURL(imageUrl);
        }
      };
    }, [file, index]);

    // Cleanup URL when component unmounts
    useEffect(() => {
      return () => {
        if (imageUrl) {
          URL.revokeObjectURL(imageUrl);
        }
      };
    }, [imageUrl]);

    if (isLoading || !imageUrl) {
      return (
        <div className="w-full h-40 bg-gray-600/30 rounded-lg mb-3 overflow-hidden flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-2 mx-auto animate-pulse">
              <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <p className="text-xs text-gray-400">Loading preview...</p>
          </div>
        </div>
      );
    }

    return (
      <div className="w-full h-40 bg-gray-600/30 rounded-lg mb-3 overflow-hidden">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageUrl}
          alt={`Thumbnail ${index + 1}`}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
          onError={() => {
            console.warn('Failed to load image preview');
            setImageUrl(null);
          }}
        />
      </div>
    );
  };

  useEffect(() => {
    // Check for File API support
    if (!window.File || !window.FileReader || !window.FileList || !window.Blob) {
      console.error('File APIs not supported in this browser');
      alert('File upload is not supported in your browser. Please use a modern browser like Chrome, Firefox, or Safari.');
      return;
    }
    
    // Track form start
    try {
      formAnalytics.trackFormStart();
    } catch (error) {
      console.warn('Analytics error:', error);
    }
  }, []); // Remove formAnalytics dependency to prevent infinite re-renders

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const startTime = performance.now();
    console.log('File change event triggered');
    
    if (e.target.files && e.target.files.length > 0) {
      // Fast file processing - avoid Array.from for better performance
      const fileList = e.target.files;
      const newFiles: File[] = [];
      const maxFiles = Math.min(fileList.length, 3);
      
      // Only process up to 3 files, avoid copying entire FileList
      for (let i = 0; i < maxFiles; i++) {
        newFiles.push(fileList[i]);
      }
      
      console.log(`Selected ${newFiles.length} files in ${(performance.now() - startTime).toFixed(1)}ms`);
      
      // Update state immediately - this is the critical path
      setFiles(newFiles);
      
      // Everything else happens async to keep UI responsive
      const scheduleWork = window.requestIdleCallback || ((cb: () => void) => setTimeout(cb, 1));
      scheduleWork(() => {
        try {
          const totalSize = newFiles.reduce((sum, file) => sum + file.size, 0);
          console.log('Files:', newFiles.map(f => `${f.name} (${(f.size/1024/1024).toFixed(1)}MB)`));
          
          // Only track analytics if hooks are available
          if (uploadAnalytics?.trackFileSelection) {
            uploadAnalytics.trackFileSelection(newFiles.length, totalSize);
          }
          if (formAnalytics?.trackFormFieldInteraction) {
            formAnalytics.trackFormFieldInteraction('file_upload', 'files_selected');
          }
        } catch (error) {
          // Silently ignore analytics errors in production
          if (process.env.NODE_ENV === 'development') {
            console.warn('Analytics error:', error);
          }
        }
      });
    } else {
      console.log('No files selected');
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

    // Check if user has unlimited access (paid plan)
    if (hasUnlimitedAccess()) {
      // Paid users can analyze unlimited thumbnails
      const planType = user?.email === 'nicvenettacci@gmail.com' ? 'Admin' : getPlanDisplayName()
      console.log(`${planType} user analyzing thumbnails`)
    } else if (isUsedUp) {
      // Free users who have used their credit
      setShowUpgradeModal(true);
      return;
    }

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
        // Increment free usage counter
        increment();
        
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

        // Store compressed image URLs for display in results - much smaller storage footprint
        console.log('Compressing images for storage...');
        const compressedImageUrls: string[] = [];
        
        try {
          for (let i = 0; i < files.length; i++) {
            const file = files[i];
            console.log(`Compressing image ${i + 1}/${files.length}: ${file.name}`);
            const compressedUrl = await compressImage(file, 300, 0.6); // Small size, decent quality
            compressedImageUrls.push(compressedUrl);
          }
          
          // Aggressive storage cleanup before saving
          console.log('Performing aggressive storage cleanup...');
          
          // Clear everything except essential data first
          const essentialData = {
            thumbnails: sessionStorage.getItem('thumbnails'),
            selectedNiche: sessionStorage.getItem('selectedNiche'),
            title: sessionStorage.getItem('title')
          };
          
          sessionStorage.clear();
          
          // Check compressed size
          const compressedJson = JSON.stringify(compressedImageUrls);
          const compressedSize = new Blob([compressedJson]).size;
          console.log(`Compressed images size: ${Math.round(compressedSize / 1024)}KB (was ~${Math.round(files.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024)}MB)`);
          
          // Restore essential data
          if (essentialData.thumbnails) sessionStorage.setItem('thumbnails', essentialData.thumbnails);
          if (essentialData.selectedNiche) sessionStorage.setItem('selectedNiche', essentialData.selectedNiche);
          if (essentialData.title) sessionStorage.setItem('title', essentialData.title);
          
          // Store compressed images
          sessionStorage.setItem('imageUrls', compressedJson);
          console.log('Successfully stored compressed images');
          
        } catch (error: any) {
          console.warn('Image compression/storage failed:', error);
          
          // Ultimate fallback - skip image storage entirely
          console.log('Skipping image storage to prevent quota issues');
          sessionStorage.removeItem('imageUrls');
        }
        
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
    <>
      <main className="min-h-screen bg-gradient-to-br from-[#0a0f25] via-[#0d1229] to-[#0a0f25] flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-between items-center mb-6">
            <Link href="/" className="text-blue-400 hover:text-blue-300 text-sm transition-colors duration-200">
              ← Back to Home
            </Link>
            {!user?.isAuthenticated && (
              <Link href="/signin" className="text-blue-400 hover:text-blue-300 text-sm transition-colors duration-200">
                Sign In
              </Link>
            )}
            {user?.isAuthenticated && (
              <div className="flex items-center gap-3">
                <div className="text-sm text-gray-300">
                  Welcome back, {user.email} ({getPlanDisplayName()})
                  {user.email === 'nicvenettacci@gmail.com' && (
                    <span className="ml-2 px-2 py-1 bg-gradient-to-r from-purple-600 to-blue-600 text-xs rounded-full">
                      ADMIN
                    </span>
                  )}
                </div>
                <button
                  onClick={signOut}
                  className="text-xs text-gray-400 hover:text-gray-300 transition-colors"
                >
                  Sign Out
                </button>
              </div>
            )}
          </div>
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
                  onClick={(e) => {
                    console.log('Label clicked');
                    e.preventDefault(); // Prevent default label behavior
                    
                    // Force trigger the file input
                    const fileInput = document.getElementById('file-upload') as HTMLInputElement;
                    if (fileInput) {
                      console.log('Triggering file input click');
                      fileInput.click();
                    } else {
                      console.error('File input not found!');
                    }
                  }}
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

            {/* Enhanced File Preview Grid - Fast Selection + Lazy Loading Previews */}
            {files.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {files.map((file, index) => (
                  <div key={`${file.name}-${file.size}`} className="relative group bg-gray-700/30 rounded-xl p-4 border border-gray-600/30 hover:border-blue-500/50 transition-all duration-200">
                    <LazyImagePreview file={file} index={index} />
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
                      {(file.size / 1024 / 1024).toFixed(1)} MB • {file.type}
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
      
      {/* Upgrade Modal */}
      <UpgradeModal 
        isOpen={showUpgradeModal} 
        onClose={() => setShowUpgradeModal(false)} 
      />
    </>
  );
}

