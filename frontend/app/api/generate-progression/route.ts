import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        // Forward to backend
        const response = await fetch('http://localhost:8000/api/generate-progression', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            throw new Error('Backend progression generation failed');
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Progression generation error:', error);
        return NextResponse.json({ error: 'Progression generation failed' }, { status: 500 });
    }
}

