import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const audio = formData.get('audio') as File;

        if (!audio) {
            return NextResponse.json({ error: 'No audio file provided' }, { status: 400 });
        }

        // Forward to backend
        const backendFormData = new FormData();
        backendFormData.append('audio', audio);

        const response = await fetch('http://localhost:8000/api/detect-chords', {
            method: 'POST',
            body: backendFormData,
        });

        if (!response.ok) {
            throw new Error('Backend chord detection failed');
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Chord detection error:', error);
        return NextResponse.json({ error: 'Chord detection failed' }, { status: 500 });
    }
}

