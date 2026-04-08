import type {
  ProcessResponse,
  CreatePdfRequest,
  HealthResponse,
  ApiError,
  ApiClientErrorCode,
} from './types';

// Default to same-origin /api so DigitalOcean ingress can route frontend and backend together.
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '/api';

export class ApiClientError extends Error {
  code: ApiClientErrorCode;

  constructor(code: ApiClientErrorCode, message?: string) {
    super(message ?? code);
    this.code = code;
    this.name = 'ApiClientError';
  }
}

export function isApiClientError(error: unknown): error is ApiClientError {
  return error instanceof ApiClientError;
}

// Helper to check if response is an error
function isApiError(data: unknown): data is ApiError {
  return typeof data === 'object' && data !== null && 'detail' in data;
}

// Parse error message from API error
export function parseErrorMessage(error: ApiError): string {
  if (typeof error.detail === 'string') {
    return error.detail;
  }
  if (Array.isArray(error.detail) && error.detail.length > 0) {
    return error.detail.map(e => e.msg).join(', ');
  }
  return '';
}

function createServerError(error: ApiError): ApiClientError {
  if (typeof error.detail === 'string') {
    if (error.detail.toLowerCase() === 'session not found') {
      return new ApiClientError('session_expired');
    }

    return new ApiClientError('server_message', error.detail);
  }

  const parsed = parseErrorMessage(error);
  if (parsed) {
    return new ApiClientError('server_message', parsed);
  }

  return new ApiClientError('unknown_error');
}

// Health check
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/`);
  if (!response.ok) {
    throw new ApiClientError('backend_unavailable');
  }
  return response.json();
}

// Process video URL
export async function processYouTubeUrl(url: string, cookiesText?: string): Promise<ProcessResponse> {
  const payload = {
    url,
    output_dir: 'data/videos/',
    ...(cookiesText ? { cookies_text: cookiesText } : {}),
  };

  const response = await fetch(`${API_BASE_URL}/process-url`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  
  const data = await response.json();
  
  if (!response.ok) {
    if (isApiError(data)) {
      throw createServerError(data);
    }
    throw new ApiClientError('process_failed');
  }
  
  return data as ProcessResponse;
}

// Process uploaded video file
export async function processUploadedFile(
  file: File,
  onProgress?: (progress: number) => void
): Promise<ProcessResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);
    
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = Math.round((event.loaded / event.total) * 100);
        onProgress(progress);
      }
    });
    
    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data as ProcessResponse);
        } catch {
          reject(new ApiClientError('parse_failed'));
        }
      } else {
        try {
          const data = JSON.parse(xhr.responseText);
          if (isApiError(data)) {
            reject(createServerError(data));
          } else {
            reject(new ApiClientError('process_failed'));
          }
        } catch {
          reject(new ApiClientError('process_failed'));
        }
      }
    });
    
    xhr.addEventListener('error', () => {
      reject(new ApiClientError('network_error'));
    });
    
    xhr.open('POST', `${API_BASE_URL}/process`);
    xhr.send(formData);
  });
}

// Create PDF from selected screenshots
export async function createPdf(request: CreatePdfRequest): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/create-pdf`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    // Try to parse error JSON
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      const data = await response.json();
      if (isApiError(data)) {
        throw createServerError(data);
      }
    }
    throw new ApiClientError('pdf_failed');
  }
  
  return response.blob();
}

// Download blob as file
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
