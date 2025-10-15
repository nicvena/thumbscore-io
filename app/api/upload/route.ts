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

    // Convert files to base64 for backend processing
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const fileName = `${sessionId}-thumb${i + 1}-${file.name}`;
      
      // Convert file to base64
      const arrayBuffer = await file.arrayBuffer();
      const base64 = Buffer.from(arrayBuffer).toString('base64');
      const dataUrl = `data:${file.type};base64,${base64}`;
      
      uploadResults.push({ 
        fileName, 
        originalName: file.name,
        dataUrl: dataUrl,
        mimeType: file.type,
        size: file.size
      });
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

