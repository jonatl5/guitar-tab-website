'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Loader2, Download, ScanLine, FileImage, CheckCircle2, Upload, FileVideo } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useI18n } from '@/lib/i18n';
import { formatFileSize } from '@/lib/helpers';

interface ProcessingStateProps {
  isYouTube: boolean;
  uploadProgress?: number;
  fileName?: string;
  fileSize?: number;
}

export function ProcessingState({ isYouTube, uploadProgress = 0, fileName, fileSize }: ProcessingStateProps) {
  const { t } = useI18n();
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

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

  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
      <CardContent className="pt-6">
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

          {/* File info for upload mode */}
          {!isYouTube && fileName && (
            <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30 border border-border/50">
              <FileVideo className="w-8 h-8 text-primary flex-shrink-0" />
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
      </CardContent>
    </Card>
  );
}
