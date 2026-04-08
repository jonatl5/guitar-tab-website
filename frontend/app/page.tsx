'use client';

import { useState, useCallback } from 'react';
import { toast } from 'sonner';
import { Header } from '@/components/header';
import { SourceInput } from '@/components/source-input';
import { ProcessingState } from '@/components/processing-state';
import { ScreenshotGallery } from '@/components/screenshot-gallery';
import { PreviewDialog } from '@/components/preview-dialog';
import { WorkflowSidebar } from '@/components/workflow-sidebar';
import { EmptyState } from '@/components/empty-state';
import { ErrorState } from '@/components/error-state';
import { ZeroResultsState } from '@/components/zero-results-state';
import { SessionExpiredState } from '@/components/session-expired-state';
import { SuccessState } from '@/components/success-state';
import {
  processYouTubeUrl,
  processUploadedFile,
  createPdf,
  downloadBlob,
  isApiClientError,
} from '@/lib/api';
import { useI18n } from '@/lib/i18n';
import type { AppState, Screenshot, SourceType } from '@/lib/types';

export default function Home() {
  const { t } = useI18n();

  const getErrorMessage = useCallback((error: unknown) => {
    if (isApiClientError(error)) {
      switch (error.code) {
        case 'backend_unavailable':
          return t('errorBackendUnavailable');
        case 'process_failed':
          return t('errorProcessFailed');
        case 'parse_failed':
          return t('errorParseResponse');
        case 'network_error':
          return t('errorNetwork');
        case 'pdf_failed':
          return t('errorCreatePdfFailed');
        case 'unknown_error':
          return t('errorUnknown');
        case 'server_message':
          return error.message;
        case 'session_expired':
          return t('toastSessionExpired');
        default:
          return t('toastError');
      }
    }

    if (error instanceof Error && error.message) {
      return error.message;
    }

    return t('toastError');
  }, [t]);

  // App state
  const [state, setState] = useState<AppState>('idle');
  const [sourceType, setSourceType] = useState<SourceType>('upload'); // Upload is now default
  const [errorMessage, setErrorMessage] = useState('');
  
  // Session data
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [screenshots, setScreenshots] = useState<Screenshot[]>([]);
  
  // Selection state
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  
  // Preview state
  const [previewScreenshot, setPreviewScreenshot] = useState<Screenshot | null>(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  
  // Upload state
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFileName, setUploadedFileName] = useState<string>('');
  const [uploadedFileSize, setUploadedFileSize] = useState<number>(0);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  
  // Video URL state for preview
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string>('');
  
  // PDF state
  const [pdfBlob, setPdfBlob] = useState<Blob | null>(null);
  
  // Session expired state
  const [sessionExpired, setSessionExpired] = useState(false);

  // Process Video URL (YouTube, Bilibili, etc.)
  const handleProcessUrl = useCallback(async (url: string, cookiesText?: string) => {
    setState('processing');
    setSourceType('youtube');
    setErrorMessage('');
    setPdfBlob(null);
    setSessionExpired(false);
    setCurrentVideoUrl(url);
    setUploadedFile(null);
    
    try {
      const response = await processYouTubeUrl(url, cookiesText);
      setSessionId(response.session_id);
      setScreenshots(response.screenshots);
      setSelectedIndices(new Set());
      
      if (response.screenshots.length === 0) {
        setState('review'); // Will show zero-results state
      } else {
        setState('review');
        toast.success(t('toastSuccess', { count: response.screenshots.length }));
      }
    } catch (error) {
      const message = getErrorMessage(error);
      setErrorMessage(message);
      setState('error');
      toast.error(message);
    }
  }, [getErrorMessage, t]);

  // Process uploaded file
  const handleProcessFile = useCallback(async (file: File) => {
    setState('processing');
    setSourceType('upload');
    setErrorMessage('');
    setUploadProgress(0);
    setUploadedFileName(file.name);
    setUploadedFileSize(file.size);
    setUploadedFile(file);
    setCurrentVideoUrl('');
    setPdfBlob(null);
    setSessionExpired(false);
    
    try {
      const response = await processUploadedFile(file, (progress) => {
        setUploadProgress(progress);
      });
      setSessionId(response.session_id);
      setScreenshots(response.screenshots);
      setSelectedIndices(new Set());
      
      if (response.screenshots.length === 0) {
        setState('review'); // Will show zero-results state
      } else {
        setState('review');
        toast.success(t('toastSuccess', { count: response.screenshots.length }));
      }
    } catch (error) {
      const message = getErrorMessage(error);
      setErrorMessage(message);
      setState('error');
      toast.error(message);
    }
  }, [getErrorMessage, t]);

  // Toggle screenshot selection
  const handleToggleSelection = useCallback((index: number) => {
    setSelectedIndices(prev => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);

  // Select all screenshots
  const handleSelectAll = useCallback(() => {
    setSelectedIndices(new Set(screenshots.map(s => s.index)));
  }, [screenshots]);

  // Clear selection
  const handleClearSelection = useCallback(() => {
    setSelectedIndices(new Set());
  }, []);

  // Open preview
  const handlePreview = useCallback((screenshot: Screenshot) => {
    setPreviewScreenshot(screenshot);
    setIsPreviewOpen(true);
  }, []);

  // Navigate preview
  const handlePreviewNavigate = useCallback((direction: 'prev' | 'next') => {
    if (!previewScreenshot) return;
    const currentIndex = screenshots.findIndex(s => s.index === previewScreenshot.index);
    const newIndex = direction === 'prev' ? currentIndex - 1 : currentIndex + 1;
    if (newIndex >= 0 && newIndex < screenshots.length) {
      setPreviewScreenshot(screenshots[newIndex]);
    }
  }, [previewScreenshot, screenshots]);

  // Create PDF
  const handleCreatePdf = useCallback(async () => {
    if (!sessionId || selectedIndices.size === 0) return;
    
    setState('creatingPdf');
    setSessionExpired(false);
    
    try {
      const blob = await createPdf({
        session_id: sessionId,
        selected_indices: Array.from(selectedIndices).sort((a, b) => a - b),
      });
      setPdfBlob(blob);
      setState('success');
      toast.success(t('toastPdfSuccess'));
    } catch (error) {
      setState('review');
      
      if (isApiClientError(error) && error.code === 'session_expired') {
        setSessionExpired(true);
        toast.error(t('toastSessionExpired'));
      } else {
        const message = getErrorMessage(error);
        toast.error(message);
      }
    }
  }, [getErrorMessage, sessionId, selectedIndices, t]);

  // Download PDF
  const handleDownloadPdf = useCallback(() => {
    if (pdfBlob) {
      downloadBlob(pdfBlob, 'guitar_tabs.pdf');
      toast.success(t('toastPdfDownloadStarted'));
    }
  }, [pdfBlob, t]);

  // Reset to initial state
  const handleReset = useCallback(() => {
    setState('idle');
    setSessionId(null);
    setScreenshots([]);
    setSelectedIndices(new Set());
    setPdfBlob(null);
    setErrorMessage('');
    setUploadProgress(0);
    setUploadedFileName('');
    setUploadedFileSize(0);
    setUploadedFile(null);
    setCurrentVideoUrl('');
    setSessionExpired(false);
  }, []);

  // Switch to Video URL tab and reset for new input
  const handleSwitchToYoutube = useCallback(() => {
    setSourceType('youtube');
    setState('idle');
    setScreenshots([]);
    setSelectedIndices(new Set());
    setSessionId(null);
    setPdfBlob(null);
    setSessionExpired(false);
    setErrorMessage('');
    setCurrentVideoUrl('');
    setUploadedFile(null);
  }, []);

  // Switch to upload tab and reset for new input
  const handleSwitchToUpload = useCallback(() => {
    setSourceType('upload');
    setState('idle');
    setScreenshots([]);
    setSelectedIndices(new Set());
    setSessionId(null);
    setPdfBlob(null);
    setSessionExpired(false);
    setErrorMessage('');
    setUploadProgress(0);
    setUploadedFileName('');
    setUploadedFileSize(0);
    setUploadedFile(null);
    setCurrentVideoUrl('');
  }, []);

  // Get current preview index info
  const previewIndex = previewScreenshot 
    ? screenshots.findIndex(s => s.index === previewScreenshot.index) 
    : -1;
  const hasPrevious = previewIndex > 0;
  const hasNext = previewIndex < screenshots.length - 1;

  // Determine what to show in main content area
  const renderMainContent = () => {
    // Source Input - show when idle or error
    if (state === 'idle' || state === 'error') {
      return (
        <>
          <SourceInput
            onProcessUrl={handleProcessUrl}
            onProcessFile={handleProcessFile}
            isProcessing={false}
            uploadProgress={0}
            sourceType={sourceType}
            onSourceTypeChange={setSourceType}
          />
          {state === 'error' && (
            <ErrorState message={errorMessage} onRetry={handleReset} />
          )}
          {state === 'idle' && (
            <EmptyState />
          )}
        </>
      );
    }

    // Processing State
    if (state === 'processing') {
      return (
        <ProcessingState 
          isYouTube={sourceType === 'youtube'} 
          uploadProgress={uploadProgress}
          fileName={uploadedFileName}
          fileSize={uploadedFileSize}
          uploadedFile={uploadedFile}
          videoUrl={currentVideoUrl}
        />
      );
    }

    // Review state with zero results
    if ((state === 'review' || state === 'creatingPdf') && screenshots.length === 0) {
      return (
        <ZeroResultsState
          onTryYoutube={handleSwitchToYoutube}
          onTryUpload={handleSwitchToUpload}
          onReset={handleReset}
        />
      );
    }

    // Session expired state
    if (sessionExpired) {
      return (
        <>
          <SessionExpiredState onReprocess={handleReset} />
          {screenshots.length > 0 && (
            <ScreenshotGallery
              screenshots={screenshots}
              selectedIndices={selectedIndices}
              onToggleSelection={handleToggleSelection}
              onSelectAll={handleSelectAll}
              onClearSelection={handleClearSelection}
              onPreview={handlePreview}
            />
          )}
        </>
      );
    }

    // Success state - show prominent success card
    if (state === 'success' && pdfBlob) {
      return (
        <>
          <SuccessState
            selectedCount={selectedIndices.size}
            onDownloadPdf={handleDownloadPdf}
            onReset={handleReset}
          />
          {screenshots.length > 0 && (
            <ScreenshotGallery
              screenshots={screenshots}
              selectedIndices={selectedIndices}
              onToggleSelection={handleToggleSelection}
              onSelectAll={handleSelectAll}
              onClearSelection={handleClearSelection}
              onPreview={handlePreview}
            />
          )}
        </>
      );
    }

    // Normal review state with results
    if ((state === 'review' || state === 'creatingPdf') && screenshots.length > 0) {
      return (
        <ScreenshotGallery
          screenshots={screenshots}
          selectedIndices={selectedIndices}
          onToggleSelection={handleToggleSelection}
          onSelectAll={handleSelectAll}
          onClearSelection={handleClearSelection}
          onPreview={handlePreview}
        />
      );
    }

    return null;
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container mx-auto px-4 py-6">
        <div className="grid lg:grid-cols-[1fr_320px] gap-6">
          {/* Main Content */}
          <div className="space-y-6">
            {renderMainContent()}
          </div>

          {/* Sidebar */}
          <aside className="lg:sticky lg:top-20 lg:self-start">
            <WorkflowSidebar
              state={state}
              selectedCount={selectedIndices.size}
              totalCount={screenshots.length}
              onCreatePdf={handleCreatePdf}
              onDownloadPdf={handleDownloadPdf}
              onReset={handleReset}
              pdfBlob={pdfBlob}
              isCreatingPdf={state === 'creatingPdf'}
            />
          </aside>
        </div>
      </main>

      {/* Preview Dialog */}
      <PreviewDialog
        screenshot={previewScreenshot}
        isOpen={isPreviewOpen}
        onClose={() => setIsPreviewOpen(false)}
        isSelected={previewScreenshot ? selectedIndices.has(previewScreenshot.index) : false}
        onToggleSelection={() => {
          if (previewScreenshot) {
            handleToggleSelection(previewScreenshot.index);
          }
        }}
        onPrevious={() => handlePreviewNavigate('prev')}
        onNext={() => handlePreviewNavigate('next')}
        hasPrevious={hasPrevious}
        hasNext={hasNext}
      />
    </div>
  );
}
