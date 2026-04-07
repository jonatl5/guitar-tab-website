'use client';

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Clock, Check, X, ChevronLeft, ChevronRight } from 'lucide-react';
import { formatTimestamp } from '@/lib/helpers';
import { useI18n } from '@/lib/i18n';
import type { Screenshot } from '@/lib/types';

interface PreviewDialogProps {
  screenshot: Screenshot | null;
  isOpen: boolean;
  onClose: () => void;
  isSelected: boolean;
  onToggleSelection: () => void;
  onPrevious?: () => void;
  onNext?: () => void;
  hasPrevious: boolean;
  hasNext: boolean;
}

export function PreviewDialog({
  screenshot,
  isOpen,
  onClose,
  isSelected,
  onToggleSelection,
  onPrevious,
  onNext,
  hasPrevious,
  hasNext,
}: PreviewDialogProps) {
  const { t } = useI18n();

  if (!screenshot) return null;

  return (
    <Dialog open={isOpen} onOpenChange={() => onClose()}>
      <DialogContent className="max-w-4xl w-[95vw] p-0 bg-card border-border/50 overflow-hidden">
        <DialogHeader className="p-4 border-b border-border/50">
          <div className="flex items-center justify-between">
            <DialogTitle className="text-base font-medium flex items-center gap-2">
              {t('tabPreview')}
              <Badge variant="secondary" className="font-mono text-xs">
                <Clock className="w-3 h-3 mr-1" />
                {formatTimestamp(screenshot.timestamp)}
              </Badge>
              <Badge variant="outline" className="font-mono text-xs text-muted-foreground">
                #{screenshot.index + 1}
              </Badge>
            </DialogTitle>
          </div>
        </DialogHeader>

        {/* Image container */}
        <div className="relative bg-secondary/30">
          <div className="aspect-[16/9] max-h-[60vh] flex items-center justify-center p-4">
            <img
              src={`data:image/png;base64,${screenshot.image}`}
              alt={`${t('tabAt')} ${formatTimestamp(screenshot.timestamp)}`}
              className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
            />
          </div>

          {/* Navigation buttons */}
          {hasPrevious && (
            <Button
              variant="secondary"
              size="icon"
              className="absolute left-4 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-background/80 backdrop-blur-sm hover:bg-background"
              onClick={onPrevious}
            >
              <ChevronLeft className="w-5 h-5" />
            </Button>
          )}
          {hasNext && (
            <Button
              variant="secondary"
              size="icon"
              className="absolute right-4 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-background/80 backdrop-blur-sm hover:bg-background"
              onClick={onNext}
            >
              <ChevronRight className="w-5 h-5" />
            </Button>
          )}
        </div>

        {/* Actions */}
        <div className="p-4 border-t border-border/50 flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            {t('timestamp')}: <span className="font-mono text-foreground">{screenshot.timestamp.toFixed(2)}s</span>
          </p>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onClose}>
              <X className="w-4 h-4 mr-2" />
              {t('close')}
            </Button>
            <Button
              variant={isSelected ? 'secondary' : 'default'}
              onClick={onToggleSelection}
              className={isSelected ? '' : 'bg-primary hover:bg-primary/90 text-primary-foreground'}
            >
              <Check className="w-4 h-4 mr-2" />
              {isSelected ? t('deselectThis') : t('selectThis')}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
