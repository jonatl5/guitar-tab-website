'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AlertTriangle, RotateCcw, Clock } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

interface SessionExpiredStateProps {
  onReprocess: () => void;
}

export function SessionExpiredState({ onReprocess }: SessionExpiredStateProps) {
  const { t } = useI18n();

  return (
    <Card className="border-amber-500/30 bg-amber-500/5">
      <CardContent className="pt-6">
        <div className="flex flex-col items-center justify-center py-6 text-center">
          <div className="w-14 h-14 rounded-full bg-amber-500/10 flex items-center justify-center mb-4">
            <AlertTriangle className="w-7 h-7 text-amber-500" />
          </div>
          <h3 className="text-lg font-medium text-foreground mb-2">
            {t('sessionExpired')}
          </h3>
          <p className="text-sm text-muted-foreground max-w-sm mb-2">
            {t('sessionExpiredDesc')}
          </p>
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground/70 mb-6">
            <Clock className="w-3.5 h-3.5" />
            <span>{t('helpSessionNote')}</span>
          </div>
          <Button 
            onClick={onReprocess} 
            className="bg-amber-600 hover:bg-amber-700 text-foreground"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            {t('reprocessVideo')}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
