import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        // Forward to backend
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            throw new Error('Backend generation failed');
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Generation error:', error);
        return NextResponse.json({ error: 'Generation failed' }, { status: 500 });
    }
}

