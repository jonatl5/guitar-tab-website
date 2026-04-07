'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AlertCircle, RotateCcw } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

interface ErrorStateProps {
  message: string;
  onRetry: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  const { t } = useI18n();

  return (
    <Card className="border-destructive/50 bg-destructive/5">
      <CardContent className="pt-6">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-14 h-14 rounded-full bg-destructive/10 flex items-center justify-center mb-4">
            <AlertCircle className="w-7 h-7 text-destructive" />
          </div>
          <h3 className="text-lg font-medium text-foreground mb-2">
            {t('processingFailed')}
          </h3>
          <p className="text-sm text-muted-foreground max-w-sm mb-6">
            {message}
          </p>
          <Button onClick={onRetry} variant="outline" className="gap-2">
            <RotateCcw className="w-4 h-4" />
            {t('retry')}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
