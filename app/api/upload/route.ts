import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'crypto';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const files: File[] = [];
    
    // Extract all uploaded files
    for (let i = 0; i < 3; i++) {
      const file = formData.get(`file${i}`) as File;
      if (file) {
        files.push(file);
      }
    }

    if (files.length === 0) {
      return NextResponse.json(
        { error: 'No files provided' },
        { status: 400 }
      );
    }

    const sessionId = randomUUID();
    const uploadResults = [];

    // Simulate file processing (in real app, upload to S3)
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const fileName = `${sessionId}-thumb${i + 1}-${file.name}`;
      
      // Simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 100));
      
      uploadResults.push({ fileName, originalName: file.name });
    }

    // Return session data for frontend to handle analysis
    return NextResponse.json({
      sessionId,
      thumbnails: uploadResults,
      message: 'Files uploaded successfully',
      status: 'ready_for_analysis'
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Upload failed' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'Upload API ready',
    status: 'operational'
  });
}

