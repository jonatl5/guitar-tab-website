'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Grid3X3, CheckSquare, Square, ZoomIn, Clock } from 'lucide-react';
import { formatTimestamp } from '@/lib/helpers';
import { useI18n } from '@/lib/i18n';
import type { Screenshot } from '@/lib/types';

interface ScreenshotGalleryProps {
  screenshots: Screenshot[];
  selectedIndices: Set<number>;
  onToggleSelection: (index: number) => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onPreview: (screenshot: Screenshot) => void;
}

export function ScreenshotGallery({
  screenshots,
  selectedIndices,
  onToggleSelection,
  onSelectAll,
  onClearSelection,
  onPreview,
}: ScreenshotGalleryProps) {
  const { t } = useI18n();

  return (
    <Card className="border-border/50 bg-card/80 backdrop-blur-sm">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Grid3X3 className="w-4 h-4 text-primary" />
            {t('extractionResults')}
            <Badge variant="secondary" className="ml-1 bg-secondary/50 text-muted-foreground">
              {screenshots.length} {t('countPhotos')}
            </Badge>
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={onSelectAll}
              className="text-xs h-7 text-muted-foreground hover:text-foreground"
            >
              <CheckSquare className="w-3.5 h-3.5 mr-1" />
              {t('selectAll')}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClearSelection}
              className="text-xs h-7 text-muted-foreground hover:text-foreground"
              disabled={selectedIndices.size === 0}
            >
              <Square className="w-3.5 h-3.5 mr-1" />
              {t('clear')}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] lg:h-[500px] pr-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {screenshots.map((screenshot) => {
              const isSelected = selectedIndices.has(screenshot.index);
              return (
                <div
                  key={screenshot.index}
                  className={`
                    group relative rounded-lg overflow-hidden cursor-pointer transition-all
                    border-2 hover:border-primary/50
                    ${isSelected 
                      ? 'border-primary ring-2 ring-primary/20' 
                      : 'border-transparent bg-secondary/30'
                    }
                  `}
                  onClick={() => onToggleSelection(screenshot.index)}
                >
                  {/* Thumbnail */}
                  <div className="aspect-[4/3] relative">
                    <img
                      src={`data:image/png;base64,${screenshot.image}`}
                      alt={`${t('tabAt')} ${formatTimestamp(screenshot.timestamp)}`}
                      className="w-full h-full object-cover"
                    />
                    
                    {/* Selection overlay */}
                    {isSelected && (
                      <div className="absolute inset-0 bg-primary/20" />
                    )}

                    {/* Hover overlay */}
                    <div className="absolute inset-0 bg-gradient-to-t from-background/90 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                    {/* Selection checkbox */}
                    <div className={`
                      absolute top-2 left-2 w-5 h-5 rounded border-2 flex items-center justify-center transition-all
                      ${isSelected 
                        ? 'bg-primary border-primary' 
                        : 'bg-background/80 border-muted-foreground/50 group-hover:border-primary/70'
                      }
                    `}>
                      {isSelected && (
                        <svg className="w-3 h-3 text-primary-foreground" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      )}
                    </div>

                    {/* Preview button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onPreview(screenshot);
                      }}
                      className="absolute top-2 right-2 w-7 h-7 rounded-md bg-background/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all hover:bg-primary hover:text-primary-foreground"
                    >
                      <ZoomIn className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Timestamp badge */}
                  <div className="absolute bottom-2 left-2 right-2">
                    <Badge 
                      variant="secondary" 
                      className="bg-background/80 backdrop-blur-sm text-xs font-mono"
                    >
                      <Clock className="w-3 h-3 mr-1" />
                      {formatTimestamp(screenshot.timestamp)}
                    </Badge>
                  </div>
                </div>
              );
            })}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
