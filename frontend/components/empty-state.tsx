'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Music, FileMusic } from 'lucide-react';
import { useI18n } from '@/lib/i18n';

export function EmptyState() {
  const { t } = useI18n();

  return (
    <Card className="border-border/50 bg-card/50">
      <CardContent className="pt-6">
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="relative mb-6">
            <div className="w-20 h-20 rounded-2xl bg-primary/10 flex items-center justify-center">
              <Music className="w-10 h-10 text-primary/60" />
            </div>
            <div className="absolute -bottom-2 -right-2 w-8 h-8 rounded-lg bg-secondary flex items-center justify-center">
              <FileMusic className="w-4 h-4 text-muted-foreground" />
            </div>
          </div>
          <h3 className="text-lg font-medium text-foreground mb-2">
            {t('readyToExtract')}
          </h3>
          <p className="text-sm text-muted-foreground max-w-xs">
            {t('emptyStateDesc')}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
