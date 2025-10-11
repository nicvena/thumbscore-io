import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { action } = await request.json();

    return NextResponse.json({
      message: 'Data collection API ready',
      action: action || 'none',
      status: 'placeholder'
    });
  } catch {
    return NextResponse.json(
      { error: 'Data collection failed' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    status: 'Data collection API ready',
    message: 'This is a placeholder endpoint'
  });
}
