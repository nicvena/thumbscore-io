'use client';

import { useEffect } from 'react';
import { initGA } from '@/lib/analytics';

interface AnalyticsProviderProps {
  children: React.ReactNode;
}

export const AnalyticsProvider = ({ children }: AnalyticsProviderProps) => {
  useEffect(() => {
    // Initialize Google Analytics
    initGA();
  }, []);

  return <>{children}</>;
};

export default AnalyticsProvider;
