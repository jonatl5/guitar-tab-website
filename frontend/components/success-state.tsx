'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { CheckCircle2, Download, RotateCcw, FileText } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

interface SuccessStateProps {
  selectedCount: number;
  onDownloadPdf: () => void;
  onReset: () => void;
}

export function SuccessState({ selectedCount, onDownloadPdf, onReset }: SuccessStateProps) {
  const { t } = useI18n();

  return (
    <Card className="border-emerald-500/30 bg-emerald-500/5">
      <CardContent className="pt-6">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          {/* Success Icon */}
          <div className="relative mb-6">
            <div className="w-20 h-20 rounded-full bg-emerald-500/20 flex items-center justify-center">
              <CheckCircle2 className="w-10 h-10 text-emerald-400" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-8 h-8 rounded-lg bg-emerald-500/30 flex items-center justify-center border border-emerald-500/50">
              <FileText className="w-4 h-4 text-emerald-400" />
            </div>
          </div>
          
          <h3 className="text-xl font-semibold text-foreground mb-2">
            {t('pdfReady')}
          </h3>
          <p className="text-sm text-muted-foreground mb-1">
            {t('pdfReadyDesc')}
          </p>
          <p className="text-sm text-emerald-400 font-medium mb-6">
            {t('pdfContains', { count: selectedCount })}
          </p>

          <div className="flex flex-col sm:flex-row gap-3 w-full max-w-xs">
            <Button 
              onClick={onDownloadPdf} 
              className="flex-1 bg-emerald-600 hover:bg-emerald-700 text-foreground"
              size="lg"
            >
              <Download className="w-4 h-4 mr-2" />
              {t('downloadPdf')}
            </Button>
          </div>
          <Button 
            onClick={onReset} 
            variant="ghost"
            className="mt-4 text-muted-foreground"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            {t('processNewVideo')}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
