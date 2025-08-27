import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const file = formData.get('file') as File;

        if (!file) {
            return NextResponse.json({ error: 'No file provided' }, { status: 400 });
        }

        // Forward to backend
        const backendFormData = new FormData();
        backendFormData.append('file', file);

        const response = await fetch('http://localhost:8000/upload/', {
            method: 'POST',
            body: backendFormData,
        });

        if (!response.ok) {
            throw new Error('Backend upload failed');
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Upload error:', error);
        return NextResponse.json({ error: 'Upload failed' }, { status: 500 });
    }
}

