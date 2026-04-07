'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Link2, 
  Loader2, 
  Grid3X3, 
  FileDown, 
  CheckCircle2, 
  Download,
  RotateCcw,
  Info
} from 'lucide-react';
import { useI18n } from '@/lib/i18n';
import type { AppState } from '@/lib/types';

interface WorkflowSidebarProps {
  state: AppState;
  selectedCount: number;
  totalCount: number;
  onCreatePdf: () => void;
  onDownloadPdf: () => void;
  onReset: () => void;
  pdfBlob: Blob | null;
  isCreatingPdf: boolean;
}

function getActiveStep(state: AppState): number {
  switch (state) {
    case 'idle':
    case 'validating':
      return 0;
    case 'processing':
      return 1;
    case 'review':
    case 'creatingPdf':
      return 2;
    case 'success':
      return 3;
    case 'error':
      return 0;
    default:
      return 0;
  }
}

export function WorkflowSidebar({
  state,
  selectedCount,
  totalCount,
  onCreatePdf,
  onDownloadPdf,
  onReset,
  pdfBlob,
  isCreatingPdf,
}: WorkflowSidebarProps) {
  const { t } = useI18n();
  const activeStep = getActiveStep(state);

  const workflowSteps = [
    { id: 'input', label: t('stepSelectVideo'), icon: Link2 },
    { id: 'process', label: t('stepExtractTabs'), icon: Loader2 },
    { id: 'review', label: t('stepSelectScreenshots'), icon: Grid3X3 },
    { id: 'download', label: t('stepDownloadPdf'), icon: FileDown },
  ];

  return (
    <div className="space-y-4">
      {/* Workflow Steps */}
      <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            {t('workflow')}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {workflowSteps.map((step, index) => {
            const Icon = step.icon;
            const isActive = index === activeStep;
            const isComplete = index < activeStep;

            return (
              <div
                key={step.id}
                className={`
                  flex items-center gap-3 p-2.5 rounded-md transition-all
                  ${isActive ? 'bg-primary/10' : ''}
                `}
              >
                <div className={`
                  w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium transition-colors
                  ${isComplete ? 'bg-emerald-500/20 text-emerald-400' : 
                    isActive ? 'bg-primary/20 text-primary' : 'bg-secondary text-muted-foreground'}
                `}>
                  {isComplete ? (
                    <CheckCircle2 className="w-4 h-4" />
                  ) : (
                    <Icon className={`w-3.5 h-3.5 ${isActive && step.id === 'process' ? 'animate-spin' : ''}`} />
                  )}
                </div>
                <span className={`
                  text-sm
                  ${isComplete ? 'text-muted-foreground' : 
                    isActive ? 'text-foreground font-medium' : 'text-muted-foreground'}
                `}>
                  {step.label}
                </span>
                {isActive && (
                  <Badge variant="secondary" className="ml-auto text-xs bg-primary/10 text-primary border-0">
                    {t('current')}
                  </Badge>
                )}
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* Selection Summary & Actions */}
      {(state === 'review' || state === 'creatingPdf' || state === 'success') && (
        <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {t('selected')}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold text-foreground">{selectedCount}</span>
              <span className="text-sm text-muted-foreground">/ {totalCount} {t('countPhotos')}</span>
            </div>
            
            {/* Progress bar */}
            <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
              <div 
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${totalCount > 0 ? (selectedCount / totalCount) * 100 : 0}%` }}
              />
            </div>

            <Separator className="bg-border/50" />

            {state === 'success' && pdfBlob ? (
              <Button
                onClick={onDownloadPdf}
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-foreground"
              >
                <Download className="w-4 h-4 mr-2" />
                {t('downloadPdf')}
              </Button>
            ) : (
              <Button
                onClick={onCreatePdf}
                disabled={selectedCount === 0 || isCreatingPdf}
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
              >
                {isCreatingPdf ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {t('creatingPdf')}
                  </>
                ) : (
                  <>
                    <FileDown className="w-4 h-4 mr-2" />
                    {t('createPdf')}
                  </>
                )}
              </Button>
            )}

            {state === 'success' && (
              <Button
                variant="outline"
                onClick={onReset}
                className="w-full"
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                {t('processNewVideo')}
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {/* Help Card */}
      <Card className="border-border/50 bg-card/50">
        <CardContent className="pt-4">
          <div className="flex gap-3">
            <Info className="w-4 h-4 text-muted-foreground flex-shrink-0 mt-0.5" />
            <div className="space-y-2 text-xs text-muted-foreground">
              <p>
                <strong className="text-foreground">{t('helpYoutubeTitle')}</strong> - {t('helpYoutubeDesc')}
              </p>
              <p>
                <strong className="text-foreground">{t('helpUploadTitle')}</strong> - {t('helpUploadDesc')}
              </p>
              <p className="text-muted-foreground/70">
                {t('helpSessionNote')}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
