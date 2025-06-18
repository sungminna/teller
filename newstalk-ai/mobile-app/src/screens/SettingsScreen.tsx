import React, { useState, useCallback } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Alert,
} from 'react-native';
import {
  List,
  Switch,
  Button,
  Divider,
  Text,
  Card,
  Chip,
  useTheme,
  Portal,
  Modal,
  RadioButton,
  Slider,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';

// Store
import { useThemeStore, ThemeMode } from '../store/themeStore';
import { useAuthStore } from '../store/authStore';
import { useNewsStore } from '../store/newsStore';

// Constants
const INTERESTS = [
  { id: 'politics', name: '정치' },
  { id: 'economy', name: '경제' },
  { id: 'society', name: '사회' },
  { id: 'culture', name: '문화' },
  { id: 'sports', name: '스포츠' },
  { id: 'tech', name: '기술' },
  { id: 'international', name: '국제' },
  { id: 'entertainment', name: '연예' },
];

const VOICE_STYLES = [
  { id: 'professional', name: '전문적', description: '뉴스 앵커 스타일' },
  { id: 'friendly', name: '친근한', description: '편안하고 친근한 톤' },
  { id: 'energetic', name: '활기찬', description: '밝고 에너지 넘치는 톤' },
  { id: 'calm', name: '차분한', description: '조용하고 안정적인 톤' },
];

export default function SettingsScreen() {
  const theme = useTheme();
  
  // Store state
  const { theme: currentTheme, mode, setMode } = useThemeStore();
  const { user, updateUser, logout } = useAuthStore();
  const { categories, updateCategories } = useNewsStore();
  
  // Local state
  const [showInterestsModal, setShowInterestsModal] = useState(false);
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const [showThemeModal, setShowThemeModal] = useState(false);
  const [selectedInterests, setSelectedInterests] = useState<string[]>(
    user?.preferences.interests || []
  );
  const [selectedVoiceStyle, setSelectedVoiceStyle] = useState(
    user?.preferences.voiceStyle || 'professional'
  );
  const [playbackSpeed, setPlaybackSpeed] = useState(
    user?.preferences.playbackSpeed || 1.0
  );
  const [notificationsEnabled, setNotificationsEnabled] = useState(
    user?.preferences.notificationsEnabled ?? true
  );
  const [offlineMode, setOfflineMode] = useState(
    user?.preferences.offlineMode ?? false
  );

  // Handle interest toggle
  const handleInterestToggle = useCallback((interestId: string) => {
    setSelectedInterests(prev => {
      if (prev.includes(interestId)) {
        return prev.filter(id => id !== interestId);
      } else {
        return [...prev, interestId];
      }
    });
  }, []);

  // Save interests
  const saveInterests = useCallback(() => {
    if (user) {
      updateUser({
        preferences: {
          ...user.preferences,
          interests: selectedInterests,
        },
      });
    }
    setShowInterestsModal(false);
  }, [user, selectedInterests, updateUser]);

  // Save voice style
  const saveVoiceStyle = useCallback((voiceId: string) => {
    if (user) {
      updateUser({
        preferences: {
          ...user.preferences,
          voiceStyle: voiceId,
        },
      });
      setSelectedVoiceStyle(voiceId);
    }
    setShowVoiceModal(false);
  }, [user, updateUser]);

  // Handle playback speed change
  const handlePlaybackSpeedChange = useCallback((speed: number) => {
    setPlaybackSpeed(speed);
    if (user) {
      updateUser({
        preferences: {
          ...user.preferences,
          playbackSpeed: speed,
        },
      });
    }
  }, [user, updateUser]);

  // Handle notifications toggle
  const handleNotificationsToggle = useCallback(() => {
    const newValue = !notificationsEnabled;
    setNotificationsEnabled(newValue);
    if (user) {
      updateUser({
        preferences: {
          ...user.preferences,
          notificationsEnabled: newValue,
        },
      });
    }
  }, [user, notificationsEnabled, updateUser]);

  // Handle offline mode toggle
  const handleOfflineModeToggle = useCallback(() => {
    const newValue = !offlineMode;
    setOfflineMode(newValue);
    if (user) {
      updateUser({
        preferences: {
          ...user.preferences,
          offlineMode: newValue,
        },
      });
    }
  }, [user, offlineMode, updateUser]);

  // Handle theme change
  const handleThemeChange = useCallback((themeMode: ThemeMode) => {
    setMode(themeMode);
    setShowThemeModal(false);
  }, [setMode]);

  // Handle logout
  const handleLogout = useCallback(() => {
    Alert.alert(
      '로그아웃',
      '정말 로그아웃하시겠습니까?',
      [
        { text: '취소', style: 'cancel' },
        { 
          text: '로그아웃', 
          style: 'destructive',
          onPress: logout 
        },
      ]
    );
  }, [logout]);

  // Get current voice style name
  const currentVoiceStyleName = VOICE_STYLES.find(
    style => style.id === selectedVoiceStyle
  )?.name || '전문적';

  // Get current theme mode name
  const getThemeModeName = (mode: ThemeMode) => {
    switch (mode) {
      case 'light': return '라이트 모드';
      case 'dark': return '다크 모드';
      case 'system': return '시스템 설정';
      default: return '시스템 설정';
    }
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* User Profile Section */}
        <Card style={styles.profileCard}>
          <Card.Content>
            <Text variant="headlineSmall" style={styles.userName}>
              {user?.name || '사용자'}
            </Text>
            <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant }}>
              {user?.email}
            </Text>
          </Card.Content>
        </Card>

        {/* Personalization Settings */}
        <Text variant="titleMedium" style={styles.sectionTitle}>
          개인화 설정
        </Text>

        <List.Item
          title="관심 분야"
          description={`${selectedInterests.length}개 선택됨`}
          left={(props) => <List.Icon {...props} icon="heart" />}
          right={(props) => <List.Icon {...props} icon="chevron-right" />}
          onPress={() => setShowInterestsModal(true)}
          style={styles.listItem}
        />

        <List.Item
          title="음성 스타일"
          description={currentVoiceStyleName}
          left={(props) => <List.Icon {...props} icon="account-voice" />}
          right={(props) => <List.Icon {...props} icon="chevron-right" />}
          onPress={() => setShowVoiceModal(true)}
          style={styles.listItem}
        />

        <Divider style={styles.divider} />

        {/* Playback Settings */}
        <Text variant="titleMedium" style={styles.sectionTitle}>
          재생 설정
        </Text>

        <View style={styles.sliderContainer}>
          <List.Item
            title="기본 재생 속도"
            description={`${playbackSpeed}x`}
            left={(props) => <List.Icon {...props} icon="speedometer" />}
            style={styles.listItemNoPress}
          />
          <View style={styles.sliderWrapper}>
            <Text variant="bodySmall">0.5x</Text>
            <Slider
              style={styles.slider}
              minimumValue={0.5}
              maximumValue={2.0}
              step={0.25}
              value={playbackSpeed}
              onValueChange={handlePlaybackSpeedChange}
              minimumTrackTintColor={theme.colors.primary}
              maximumTrackTintColor={theme.colors.outline}
              thumbStyle={{ backgroundColor: theme.colors.primary }}
            />
            <Text variant="bodySmall">2.0x</Text>
          </View>
        </View>

        <List.Item
          title="알림"
          description="새 뉴스 및 처리 완료 알림"
          left={(props) => <List.Icon {...props} icon="bell" />}
          right={() => (
            <Switch
              value={notificationsEnabled}
              onValueChange={handleNotificationsToggle}
            />
          )}
          style={styles.listItem}
        />

        <List.Item
          title="오프라인 모드"
          description="뉴스 자동 다운로드 (24시간 캐시)"
          left={(props) => <List.Icon {...props} icon="download" />}
          right={() => (
            <Switch
              value={offlineMode}
              onValueChange={handleOfflineModeToggle}
            />
          )}
          style={styles.listItem}
        />

        <Divider style={styles.divider} />

        {/* App Settings */}
        <Text variant="titleMedium" style={styles.sectionTitle}>
          앱 설정
        </Text>

        <List.Item
          title="테마"
          description={getThemeModeName(mode)}
          left={(props) => <List.Icon {...props} icon="palette" />}
          right={(props) => <List.Icon {...props} icon="chevron-right" />}
          onPress={() => setShowThemeModal(true)}
          style={styles.listItem}
        />

        <List.Item
          title="로그아웃"
          left={(props) => <List.Icon {...props} icon="logout" />}
          right={(props) => <List.Icon {...props} icon="chevron-right" />}
          onPress={handleLogout}
          style={styles.listItem}
        />

        {/* App Info */}
        <View style={styles.appInfo}>
          <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant, textAlign: 'center' }}>
            NewsTeam AI v1.0.0
          </Text>
        </View>
      </ScrollView>

      {/* Interests Modal */}
      <Portal>
        <Modal
          visible={showInterestsModal}
          onDismiss={() => setShowInterestsModal(false)}
          contentContainerStyle={[
            styles.modalContainer,
            { backgroundColor: theme.colors.surface }
          ]}
        >
          <Text variant="headlineSmall" style={styles.modalTitle}>
            관심 분야 선택
          </Text>
          <ScrollView style={styles.modalContent}>
            <View style={styles.interestsGrid}>
              {INTERESTS.map((interest) => (
                <Chip
                  key={interest.id}
                  selected={selectedInterests.includes(interest.id)}
                  onPress={() => handleInterestToggle(interest.id)}
                  style={styles.interestChip}
                >
                  {interest.name}
                </Chip>
              ))}
            </View>
          </ScrollView>
          <View style={styles.modalActions}>
            <Button
              mode="text"
              onPress={() => setShowInterestsModal(false)}
            >
              취소
            </Button>
            <Button
              mode="contained"
              onPress={saveInterests}
            >
              저장
            </Button>
          </View>
        </Modal>
      </Portal>

      {/* Voice Style Modal */}
      <Portal>
        <Modal
          visible={showVoiceModal}
          onDismiss={() => setShowVoiceModal(false)}
          contentContainerStyle={[
            styles.modalContainer,
            { backgroundColor: theme.colors.surface }
          ]}
        >
          <Text variant="headlineSmall" style={styles.modalTitle}>
            음성 스타일 선택
          </Text>
          <ScrollView style={styles.modalContent}>
            <RadioButton.Group
              onValueChange={saveVoiceStyle}
              value={selectedVoiceStyle}
            >
              {VOICE_STYLES.map((style) => (
                <View key={style.id} style={styles.radioItem}>
                  <RadioButton.Item
                    label={style.name}
                    value={style.id}
                    style={styles.radioButton}
                  />
                  <Text
                    variant="bodySmall"
                    style={[
                      styles.voiceDescription,
                      { color: theme.colors.onSurfaceVariant }
                    ]}
                  >
                    {style.description}
                  </Text>
                </View>
              ))}
            </RadioButton.Group>
          </ScrollView>
          <Button
            mode="text"
            onPress={() => setShowVoiceModal(false)}
            style={styles.modalCloseButton}
          >
            닫기
          </Button>
        </Modal>
      </Portal>

      {/* Theme Modal */}
      <Portal>
        <Modal
          visible={showThemeModal}
          onDismiss={() => setShowThemeModal(false)}
          contentContainerStyle={[
            styles.modalContainer,
            { backgroundColor: theme.colors.surface }
          ]}
        >
          <Text variant="headlineSmall" style={styles.modalTitle}>
            테마 선택
          </Text>
          <RadioButton.Group
            onValueChange={handleThemeChange}
            value={mode}
          >
            <RadioButton.Item
              label="라이트 모드"
              value="light"
              style={styles.radioButton}
            />
            <RadioButton.Item
              label="다크 모드"
              value="dark"
              style={styles.radioButton}
            />
            <RadioButton.Item
              label="시스템 설정"
              value="system"
              style={styles.radioButton}
            />
          </RadioButton.Group>
          <Button
            mode="text"
            onPress={() => setShowThemeModal(false)}
            style={styles.modalCloseButton}
          >
            닫기
          </Button>
        </Modal>
      </Portal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  profileCard: {
    margin: 16,
    marginBottom: 8,
  },
  userName: {
    marginBottom: 4,
  },
  sectionTitle: {
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  listItem: {
    paddingHorizontal: 16,
  },
  listItemNoPress: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  divider: {
    marginVertical: 8,
  },
  sliderContainer: {
    marginBottom: 8,
  },
  sliderWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingBottom: 8,
  },
  slider: {
    flex: 1,
    marginHorizontal: 12,
  },
  appInfo: {
    padding: 20,
    marginTop: 20,
  },
  modalContainer: {
    margin: 20,
    padding: 20,
    borderRadius: 12,
    maxHeight: '80%',
  },
  modalTitle: {
    marginBottom: 16,
    textAlign: 'center',
  },
  modalContent: {
    maxHeight: 400,
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  modalCloseButton: {
    marginTop: 16,
    alignSelf: 'center',
  },
  interestsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  interestChip: {
    marginVertical: 2,
  },
  radioItem: {
    marginBottom: 8,
  },
  radioButton: {
    paddingHorizontal: 0,
  },
  voiceDescription: {
    marginLeft: 56,
    marginTop: -4,
    marginBottom: 8,
  },
}); 