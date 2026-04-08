'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { 
  Loader2, Download, ScanLine, FileImage, CheckCircle2, 
  Upload, FileVideo, Play, Monitor
} from 'lucide-react';
import { useEffect, useState, useRef, useMemo } from 'react';
import { useI18n } from '@/lib/i18n';
import { formatFileSize, extractYouTubeVideoId, getYouTubeThumbnail, getVideoPlatformLabel } from '@/lib/helpers';

interface ProcessingStateProps {
  isYouTube: boolean;
  uploadProgress?: number;
  fileName?: string;
  fileSize?: number;
  // Future-ready props for video preview
  uploadedFile?: File | null;
  videoUrl?: string;
  previewVideoUrl?: string | null;
  previewEmbedUrl?: string | null;
  previewThumbnailUrl?: string | null;
  sourceLabel?: string;
}

export function ProcessingState({ 
  isYouTube, 
  uploadProgress = 0, 
  fileName, 
  fileSize,
  uploadedFile,
  videoUrl,
  previewVideoUrl,
  previewEmbedUrl,
  previewThumbnailUrl,
  sourceLabel,
}: ProcessingStateProps) {
  const { t } = useI18n();
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  // Create object URL for uploaded file
  const localVideoUrl = useMemo(() => {
    if (uploadedFile && !isYouTube) {
      return URL.createObjectURL(uploadedFile);
    }
    return null;
  }, [uploadedFile, isYouTube]);

  // Cleanup object URL on unmount
  useEffect(() => {
    return () => {
      if (localVideoUrl) {
        URL.revokeObjectURL(localVideoUrl);
      }
    };
  }, [localVideoUrl]);

  // YouTube video ID for thumbnail
  const youtubeVideoId = useMemo(() => {
    if (isYouTube && videoUrl) {
      return extractYouTubeVideoId(videoUrl);
    }
    return null;
  }, [isYouTube, videoUrl]);

  // Steps for YouTube processing
  const youtubeSteps = [
    { id: 'download', label: t('stepDownload'), icon: Download },
    { id: 'analyze', label: t('stepAnalyze'), icon: ScanLine },
    { id: 'extract', label: t('stepExtract'), icon: FileImage },
  ];

  // Steps for upload processing
  const uploadSteps = [
    { id: 'upload', label: t('stepUploading'), icon: Upload },
    { id: 'analyze', label: t('stepAnalyze'), icon: ScanLine },
    { id: 'extract', label: t('stepExtract'), icon: FileImage },
  ];

  const steps = isYouTube ? youtubeSteps : uploadSteps;

  useEffect(() => {
    if (isYouTube) {
      // Simulate indeterminate progress for YouTube processing
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 2;
        });
      }, 200);

      const stepInterval = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) return prev;
          return prev + 1;
        });
      }, 5000);

      return () => {
        clearInterval(progressInterval);
        clearInterval(stepInterval);
      };
    } else {
      // For upload, use real upload progress to determine step
      if (uploadProgress < 100) {
        setCurrentStep(0);
      } else {
        // Upload complete, now processing
        setCurrentStep(1);
        // Simulate analysis step progress
        const analysisInterval = setInterval(() => {
          setCurrentStep(prev => {
            if (prev >= steps.length - 1) return prev;
            return prev + 1;
          });
        }, 4000);
        return () => clearInterval(analysisInterval);
      }
    }
  }, [isYouTube, uploadProgress, steps.length]);

  // Determine video preview content
  const renderVideoPreview = () => {
    // For upload mode with actual file - show HTML5 video player
    if (!isYouTube && localVideoUrl) {
      return (
        <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
          <div className="aspect-video relative">
            <video
              ref={videoRef}
              src={localVideoUrl}
              className="w-full h-full object-contain bg-black"
              controls
              muted
              playsInline
              preload="metadata"
            />
          </div>
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-background/90 to-transparent p-3">
            <div className="flex items-center gap-2">
              <Play className="w-4 h-4 text-primary" />
              <span className="text-xs text-foreground font-medium">{t('playbackReady')}</span>
            </div>
          </div>
        </div>
      );
    }

    // Future-ready: If we have a preview video URL (direct playable URL)
    if (previewVideoUrl) {
      return (
        <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
          <div className="aspect-video relative">
            <video
              src={previewVideoUrl}
              className="w-full h-full object-contain bg-black"
              controls
              muted
              playsInline
            />
          </div>
        </div>
      );
    }

    // Future-ready: If we have an embeddable preview (e.g., YouTube embed)
    if (previewEmbedUrl) {
      return (
        <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
          <div className="aspect-video relative">
            <iframe
              src={previewEmbedUrl}
              className="w-full h-full"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </div>
      );
    }

    // For YouTube URLs - show thumbnail preview
    if (isYouTube && youtubeVideoId) {
      return (
        <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
          <div className="aspect-video relative">
            <img
              src={previewThumbnailUrl || getYouTubeThumbnail(youtubeVideoId)}
              alt={t('videoThumbnailAlt')}
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center border border-primary/30 backdrop-blur-sm">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
              </div>
            </div>
            <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
              <span className="text-xs text-foreground font-medium bg-background/60 px-2 py-1 rounded backdrop-blur-sm">
                YouTube
              </span>
              <span className="text-xs text-muted-foreground bg-background/60 px-2 py-1 rounded backdrop-blur-sm">
                {t('watchWhileProcessing')}
              </span>
            </div>
          </div>
        </div>
      );
    }

    // For non-YouTube URLs - show platform-aware placeholder
    if (isYouTube && videoUrl) {
      const platformLabel = sourceLabel || getVideoPlatformLabel(videoUrl);
      return (
        <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
          <div className="aspect-video relative">
            {previewThumbnailUrl ? (
              <img
                src={previewThumbnailUrl}
                alt={t('videoThumbnailAlt')}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-secondary via-secondary/80 to-secondary/60 flex flex-col items-center justify-center">
                <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-4 border border-primary/20">
                  <Monitor className="w-10 h-10 text-primary/70" />
                </div>
                <span className="text-sm text-foreground font-medium mb-1">{platformLabel}</span>
                <span className="text-xs text-muted-foreground">{t('previewNotAvailable')}</span>
              </div>
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-transparent" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center border border-primary/30 backdrop-blur-sm">
                <Loader2 className="w-8 h-8 text-primary animate-spin" />
              </div>
            </div>
            <div className="absolute bottom-3 left-3">
              <span className="text-xs text-foreground font-medium bg-background/60 px-2 py-1 rounded backdrop-blur-sm">
                {platformLabel}
              </span>
            </div>
          </div>
        </div>
      );
    }

    // Fallback: generic processing placeholder
    return (
      <div className="relative rounded-lg overflow-hidden border border-border/50 bg-secondary/30">
        <div className="aspect-video relative">
          <div className="w-full h-full bg-gradient-to-br from-secondary via-secondary/80 to-secondary/60 flex flex-col items-center justify-center">
            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-4 border border-primary/20">
              <FileVideo className="w-10 h-10 text-primary/70" />
            </div>
            <span className="text-sm text-muted-foreground">{t('videoPreview')}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
      <CardContent className="pt-6">
        {/* Workstation Layout - Video on left, progress on right (stacked on mobile) */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left: Video Preview Panel */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 mb-2">
              <Play className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium text-foreground">{t('videoPreview')}</span>
            </div>
            {renderVideoPreview()}
            
            {/* File info for upload mode */}
            {!isYouTube && fileName && (
              <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30 border border-border/50">
                <FileVideo className="w-6 h-6 text-primary flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{fileName}</p>
                  {fileSize && (
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(fileSize)}
                    </p>
                  )}
                </div>
                {uploadProgress < 100 && (
                  <span className="text-sm font-mono text-primary">{uploadProgress}%</span>
                )}
              </div>
            )}
          </div>

          {/* Right: Processing Status */}
          <div className="space-y-6">
            {/* Main loading indicator */}
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="relative">
                <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                  <Loader2 className="w-8 h-8 text-primary animate-spin" />
                </div>
                <div className="absolute -inset-1 rounded-full bg-primary/20 animate-ping" />
              </div>
              <div>
                <h3 className="text-lg font-medium text-foreground">{t('processingVideo')}</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  {isYouTube ? t('downloadingFromYoutube') : t('analyzingUploadedVideo')}
                </p>
              </div>
            </div>

            {/* Progress bar */}
            {!isYouTube && uploadProgress < 100 ? (
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">{t('uploadProgress')}</span>
                  <span className="text-primary font-mono">{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-1.5" />
              </div>
            ) : isYouTube ? (
              <div className="space-y-2">
                <Progress value={progress} className="h-1.5" />
                <p className="text-xs text-center text-muted-foreground">
                  {t('processingTimeNote')}
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <Progress value={100} className="h-1.5" />
                <p className="text-xs text-center text-muted-foreground">
                  {t('processingTimeNote')}
                </p>
              </div>
            )}

            {/* Processing steps */}
            <div className="space-y-3">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isActive = index === currentStep;
                const isComplete = index < currentStep;

                return (
                  <div
                    key={step.id}
                    className={`
                      flex items-center gap-3 p-3 rounded-lg transition-all
                      ${isActive ? 'bg-primary/10 border border-primary/30' : 'bg-secondary/30'}
                    `}
                  >
                    <div className={`
                      w-8 h-8 rounded-full flex items-center justify-center transition-colors
                      ${isComplete ? 'bg-emerald-500/20 text-emerald-400' : 
                        isActive ? 'bg-primary/20 text-primary' : 'bg-secondary text-muted-foreground'}
                    `}>
                      {isComplete ? (
                        <CheckCircle2 className="w-4 h-4" />
                      ) : isActive ? (
                        <Icon className="w-4 h-4 animate-pulse" />
                      ) : (
                        <Icon className="w-4 h-4" />
                      )}
                    </div>
                    <span className={`
                      text-sm font-medium
                      ${isComplete ? 'text-emerald-400' : 
                        isActive ? 'text-foreground' : 'text-muted-foreground'}
                    `}>
                      {step.label}
                    </span>
                    {isActive && (
                      <Loader2 className="w-4 h-4 text-primary animate-spin ml-auto" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
