'use client';

import { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Music, Wifi, WifiOff } from 'lucide-react';
import { checkHealth } from '@/lib/api';
import { useI18n } from '@/lib/i18n';

export function Header() {
  const { locale, setLocale, t } = useI18n();
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [version, setVersion] = useState<string>('');

  useEffect(() => {
    async function check() {
      try {
        const health = await checkHealth();
        setIsConnected(health.status === 'running');
        setVersion(health.version);
      } catch {
        setIsConnected(false);
      }
    }
    
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  const toggleLocale = () => {
    setLocale(locale === 'en' ? 'zh' : 'en');
  };

  return (
    <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-primary/10 border border-primary/20">
              <Music className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-foreground">
                {t('appTitle')}
              </h1>
              <p className="text-xs text-muted-foreground">
                {t('appSubtitle')}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Language Toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleLocale}
              className="h-8 px-2.5 text-xs font-medium text-muted-foreground hover:text-foreground border border-border/50 hover:border-primary/30 bg-secondary/30"
            >
              <span className={locale === 'en' ? 'text-foreground' : 'opacity-50'}>EN</span>
              <span className="mx-1.5 text-border">/</span>
              <span className={locale === 'zh' ? 'text-foreground' : 'opacity-50'}>中文</span>
            </Button>

            {/* Backend Status */}
            {isConnected === null ? (
              <Badge variant="outline" className="text-muted-foreground border-border">
                <span className="w-2 h-2 rounded-full bg-muted-foreground/50 mr-1.5 animate-pulse" />
                {t('connecting')}
              </Badge>
            ) : isConnected ? (
              <Badge variant="outline" className="text-emerald-400 border-emerald-500/30 bg-emerald-500/10">
                <Wifi className="w-3 h-3 mr-1.5" />
                {t('connected')} {version && `v${version}`}
              </Badge>
            ) : (
              <Badge variant="outline" className="text-destructive border-destructive/30 bg-destructive/10">
                <WifiOff className="w-3 h-3 mr-1.5" />
                {t('backendOffline')}
              </Badge>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
