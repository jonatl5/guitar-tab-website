// API Types for Guitar Tab Extractor

export type Screenshot = {
  index: number;
  image: string;
  timestamp: number;
};

export type ProcessResponse = {
  session_id: string;
  screenshots: Screenshot[];
  video_path?: string;
};

export type CreatePdfRequest = {
  session_id: string;
  selected_indices: number[];
};

export type HealthResponse = {
  message: string;
  version: string;
  endpoints: {
    docs: string;
    process: string;
    process_url: string;
    create_pdf: string;
  };
  status: string;
};

export type ApiError =
  | { detail: string }
  | { detail: { loc: (string | number)[]; msg: string; type: string }[] };

export type ApiClientErrorCode =
  | 'server_message'
  | 'unknown_error'
  | 'backend_unavailable'
  | 'process_failed'
  | 'parse_failed'
  | 'network_error'
  | 'pdf_failed'
  | 'session_expired';

// UI State Types
export type AppState = 
  | 'idle'
  | 'validating'
  | 'processing'
  | 'review'
  | 'creatingPdf'
  | 'success'
  | 'error';

export type SourceType = 'youtube' | 'upload';
