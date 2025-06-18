import React, { createContext, useContext, useState, useEffect } from 'react';
import { AccessibilityInfo, Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface AccessibilitySettings {
  screenReaderEnabled: boolean;
  highContrastMode: boolean;
  largeFontSize: boolean;
  reduceMotion: boolean;
  voiceAnnouncementsEnabled: boolean;
  keyboardNavigationEnabled: boolean;
  hapticFeedbackEnabled: boolean;
}

interface AccessibilityContextType {
  settings: AccessibilitySettings;
  updateSetting: (key: keyof AccessibilitySettings, value: boolean) => void;
  announceForScreenReader: (message: string) => void;
  isScreenReaderEnabled: boolean;
  fontScale: number;
}

const defaultSettings: AccessibilitySettings = {
  screenReaderEnabled: false,
  highContrastMode: false,
  largeFontSize: false,
  reduceMotion: false,
  voiceAnnouncementsEnabled: true,
  keyboardNavigationEnabled: true,
  hapticFeedbackEnabled: true,
};

const AccessibilityContext = createContext<AccessibilityContextType>({
  settings: defaultSettings,
  updateSetting: () => {},
  announceForScreenReader: () => {},
  isScreenReaderEnabled: false,
  fontScale: 1.0,
});

export const useAccessibility = () => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within AccessibilityProvider');
  }
  return context;
};

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const [settings, setSettings] = useState<AccessibilitySettings>(defaultSettings);
  const [isScreenReaderEnabled, setIsScreenReaderEnabled] = useState(false);
  const [fontScale, setFontScale] = useState(1.0);

  useEffect(() => {
    initializeAccessibility();
    setupAccessibilityListeners();
  }, []);

  const initializeAccessibility = async () => {
    try {
      // 저장된 접근성 설정 로드
      const savedSettings = await AsyncStorage.getItem('accessibility_settings');
      if (savedSettings) {
        setSettings(JSON.parse(savedSettings));
      }

      // 시스템 접근성 상태 확인
      const screenReaderEnabled = await AccessibilityInfo.isScreenReaderEnabled();
      setIsScreenReaderEnabled(screenReaderEnabled);

      // 시스템 폰트 크기 확인
      if (Platform.OS === 'ios') {
        // iOS에서는 Dynamic Type 설정을 확인
        try {
          const isReduceMotionEnabled = await AccessibilityInfo.isReduceMotionEnabled();
          setSettings(prev => ({ ...prev, reduceMotion: isReduceMotionEnabled }));
        } catch (error) {
          console.log('폰트 크기 확인 실패:', error);
        }
      }
    } catch (error) {
      console.error('접근성 초기화 실패:', error);
    }
  };

  const setupAccessibilityListeners = () => {
    // 스크린 리더 상태 변경 감지
    const screenReaderListener = AccessibilityInfo.addEventListener(
      'screenReaderChanged',
      (enabled: boolean) => {
        setIsScreenReaderEnabled(enabled);
        updateSetting('screenReaderEnabled', enabled);
      }
    );

    // 폰트 크기 변경 감지 (iOS)
    if (Platform.OS === 'ios') {
      const fontScaleListener = AccessibilityInfo.addEventListener(
        'reduceMotionChanged',
        (enabled: boolean) => {
          updateSetting('reduceMotion', enabled);
        }
      );
      
      return () => {
        screenReaderListener?.remove();
        fontScaleListener?.remove();
      };
    }

    return () => {
      screenReaderListener?.remove();
    };
  };

  const updateSetting = async (key: keyof AccessibilitySettings, value: boolean) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    
    try {
      await AsyncStorage.setItem('accessibility_settings', JSON.stringify(newSettings));
    } catch (error) {
      console.error('접근성 설정 저장 실패:', error);
    }
  };

  const announceForScreenReader = (message: string) => {
    if (isScreenReaderEnabled && settings.voiceAnnouncementsEnabled) {
      AccessibilityInfo.announceForAccessibility(message);
    }
  };

  const value: AccessibilityContextType = {
    settings,
    updateSetting,
    announceForScreenReader,
    isScreenReaderEnabled,
    fontScale,
  };

  return (
    <AccessibilityContext.Provider value={value}>
      {children}
    </AccessibilityContext.Provider>
  );
};

// 접근성 향상 HOC
export const withAccessibility = <P extends object>(
  Component: React.ComponentType<P>
) => {
  return (props: P) => {
    const accessibility = useAccessibility();
    return <Component {...props} accessibility={accessibility} />;
  };
};

// 접근성 유틸리티 함수들
export const AccessibilityUtils = {
  // 적절한 터치 영역 크기 확인 (최소 44x44dp)
  getMinimumTouchableSize: () => ({ width: 44, height: 44 }),
  
  // 색상 대비 확인 (WCAG AA 기준 4.5:1)
  checkColorContrast: (foreground: string, background: string): boolean => {
    // 간단한 색상 대비 계산 (실제로는 더 정확한 라이브러리 사용 권장)
    const getLuminance = (color: string): number => {
      // RGB 값 추출 및 상대 휘도 계산
      const rgb = color.match(/\d+/g);
      if (!rgb || rgb.length < 3) return 0;
      
      const [r, g, b] = rgb.map(val => {
        const num = parseInt(val) / 255;
        return num <= 0.03928 ? num / 12.92 : Math.pow((num + 0.055) / 1.055, 2.4);
      });
      
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    
    const l1 = getLuminance(foreground);
    const l2 = getLuminance(background);
    const contrast = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    
    return contrast >= 4.5; // WCAG AA 기준
  },
  
  // 접근성 라벨 생성
  createAccessibilityLabel: (title: string, description?: string, state?: string): string => {
    let label = title;
    if (description) label += `, ${description}`;
    if (state) label += `, ${state}`;
    return label;
  },
  
  // 키보드 탐색을 위한 포커스 순서 관리
  getFocusOrder: (elements: string[]): { [key: string]: number } => {
    return elements.reduce((order, element, index) => {
      order[element] = index + 1;
      return order;
    }, {} as { [key: string]: number });
  },
};

// 접근성 테스트 도구
export const AccessibilityTester = {
  // 컴포넌트의 접근성 검사
  testComponent: (componentProps: any): string[] => {
    const issues: string[] = [];
    
    // 접근성 라벨 확인
    if (!componentProps.accessibilityLabel && !componentProps.children) {
      issues.push('접근성 라벨이 없습니다');
    }
    
    // 터치 영역 크기 확인
    if (componentProps.style?.width && componentProps.style?.width < 44) {
      issues.push('터치 영역이 너무 작습니다 (최소 44x44dp 필요)');
    }
    
    // 역할 정의 확인
    if (componentProps.onPress && !componentProps.accessibilityRole) {
      issues.push('버튼 역할이 정의되지 않았습니다');
    }
    
    return issues;
  },
  
  // 전체 화면의 접근성 점수 계산
  calculateAccessibilityScore: (screenElements: any[]): number => {
    let totalScore = 0;
    let maxScore = screenElements.length * 100;
    
    screenElements.forEach(element => {
      const issues = AccessibilityTester.testComponent(element);
      const elementScore = Math.max(0, 100 - (issues.length * 25));
      totalScore += elementScore;
    });
    
    return maxScore > 0 ? Math.round((totalScore / maxScore) * 100) : 100;
  },
};

export default AccessibilityProvider; 