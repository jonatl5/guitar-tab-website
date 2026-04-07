'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { SearchX, Youtube, Upload, RotateCcw } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

interface ZeroResultsStateProps {
  onTryYoutube: () => void;
  onTryUpload: () => void;
  onReset: () => void;
}

export function ZeroResultsState({ onTryYoutube, onTryUpload, onReset }: ZeroResultsStateProps) {
  const { t } = useI18n();

  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
      <CardContent className="pt-6">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4">
            <SearchX className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium text-foreground mb-2">
            {t('noTabsDetected')}
          </h3>
          <p className="text-sm text-muted-foreground max-w-sm mb-4">
            {t('zeroResultsDesc')}
          </p>
          
          <ul className="text-sm text-muted-foreground text-left space-y-1.5 mb-6">
            <li className="flex items-start gap-2">
              <span className="text-primary mt-0.5">•</span>
              {t('zeroResultsReason1')}
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary mt-0.5">•</span>
              {t('zeroResultsReason2')}
            </li>
            <li className="flex items-start gap-2">
              <span className="text-primary mt-0.5">•</span>
              {t('zeroResultsReason3')}
            </li>
          </ul>

          <div className="flex flex-col sm:flex-row gap-2 w-full max-w-sm">
            <Button 
              onClick={onTryYoutube} 
              variant="default"
              className="flex-1 bg-primary hover:bg-primary/90 text-primary-foreground"
            >
              <Youtube className="w-4 h-4 mr-2" />
              {t('tryAnotherVideo')}
            </Button>
            <Button 
              onClick={onTryUpload} 
              variant="outline"
              className="flex-1"
            >
              <Upload className="w-4 h-4 mr-2" />
              {t('uploadLocalVideo')}
            </Button>
          </div>
          <Button 
            onClick={onReset} 
            variant="ghost"
            className="mt-3 text-muted-foreground"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            {t('startOver')}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
