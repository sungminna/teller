import React from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import {
  Text,
  Card,
  List,
  Avatar,
  useTheme,
  ProgressBar,
  Chip,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';

// Store
import { useAuthStore } from '../store/authStore';
import { useNewsStore } from '../store/newsStore';

export default function ProfileScreen() {
  const theme = useTheme();
  const { user } = useAuthStore();
  const { news } = useNewsStore();

  // Calculate statistics
  const totalNewsProcessed = news.filter(item => item.processingStatus === 'completed').length;
  const totalListeningTime = news.reduce((total, item) => total + (item.duration || 0), 0);
  const averageQuality = news.length > 0 
    ? news.reduce((total, item) => total + item.quality, 0) / news.length 
    : 0;

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}시간 ${minutes}분`;
    }
    return `${minutes}분`;
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Profile Header */}
        <Card style={styles.profileCard}>
          <Card.Content style={styles.profileContent}>
            <View style={styles.profileHeader}>
              <Avatar.Text
                size={80}
                label={getInitials(user?.name || 'User')}
                style={{ backgroundColor: theme.colors.primary }}
              />
              <View style={styles.profileInfo}>
                <Text variant="headlineSmall" style={styles.userName}>
                  {user?.name || '사용자'}
                </Text>
                <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant }}>
                  {user?.email}
                </Text>
                <View style={styles.interestsContainer}>
                  {user?.preferences.interests.slice(0, 3).map((interest) => (
                    <Chip key={interest} style={styles.interestChip} compact>
                      {interest}
                    </Chip>
                  ))}
                  {user?.preferences.interests.length > 3 && (
                    <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant }}>
                      +{user.preferences.interests.length - 3}개 더
                    </Text>
                  )}
                </View>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Statistics */}
        <Text variant="titleMedium" style={styles.sectionTitle}>
          이용 통계
        </Text>

        <Card style={styles.statsCard}>
          <Card.Content>
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text variant="headlineMedium" style={[styles.statValue, { color: theme.colors.primary }]}>
                  {totalNewsProcessed}
                </Text>
                <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant }}>
                  처리된 뉴스
                </Text>
              </View>
              
              <View style={styles.statItem}>
                <Text variant="headlineMedium" style={[styles.statValue, { color: theme.colors.secondary }]}>
                  {formatTime(totalListeningTime)}
                </Text>
                <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant }}>
                  총 청취 시간
                </Text>
              </View>
            </View>

            <View style={styles.qualitySection}>
              <View style={styles.qualityHeader}>
                <Text variant="titleSmall">평균 품질 점수</Text>
                <Text variant="titleSmall" style={{ color: theme.colors.primary }}>
                  {Math.round(averageQuality * 100)}%
                </Text>
              </View>
              <ProgressBar
                progress={averageQuality}
                color={theme.colors.primary}
                style={styles.qualityBar}
              />
            </View>
          </Card.Content>
        </Card>

        {/* Preferences */}
        <Text variant="titleMedium" style={styles.sectionTitle}>
          설정 정보
        </Text>

        <Card style={styles.preferencesCard}>
          <List.Item
            title="음성 스타일"
            description={user?.preferences.voiceStyle || '전문적'}
            left={(props) => <List.Icon {...props} icon="account-voice" />}
          />
          
          <List.Item
            title="기본 재생 속도"
            description={`${user?.preferences.playbackSpeed || 1.0}x`}
            left={(props) => <List.Icon {...props} icon="speedometer" />}
          />
          
          <List.Item
            title="알림"
            description={user?.preferences.notificationsEnabled ? '켜짐' : '꺼짐'}
            left={(props) => <List.Icon {...props} icon="bell" />}
          />
          
          <List.Item
            title="오프라인 모드"
            description={user?.preferences.offlineMode ? '켜짐' : '꺼짐'}
            left={(props) => <List.Icon {...props} icon="download" />}
          />
        </Card>

        {/* Recent Activity */}
        <Text variant="titleMedium" style={styles.sectionTitle}>
          최근 활동
        </Text>

        <Card style={styles.activityCard}>
          <Card.Content>
            {news.slice(0, 5).map((item) => (
              <View key={item.id} style={styles.activityItem}>
                <View style={styles.activityInfo}>
                  <Text variant="bodyMedium" numberOfLines={1}>
                    {item.title}
                  </Text>
                  <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant }}>
                    {item.category} • {item.processingStatus === 'completed' ? '완료' : '처리 중'}
                  </Text>
                </View>
                <Chip
                  style={[
                    styles.activityStatus,
                    {
                      backgroundColor: item.processingStatus === 'completed'
                        ? theme.colors.primaryContainer
                        : theme.colors.surfaceVariant
                    }
                  ]}
                  textStyle={{
                    color: item.processingStatus === 'completed'
                      ? theme.colors.onPrimaryContainer
                      : theme.colors.onSurfaceVariant
                  }}
                  compact
                >
                  {item.processingStatus === 'completed' ? '완료' : '대기'}
                </Chip>
              </View>
            ))}
            
            {news.length === 0 && (
              <Text variant="bodyMedium" style={[styles.emptyText, { color: theme.colors.onSurfaceVariant }]}>
                아직 뉴스 활동이 없습니다.
              </Text>
            )}
          </Card.Content>
        </Card>

        {/* App Info */}
        <View style={styles.appInfo}>
          <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant, textAlign: 'center' }}>
            NewsTeam AI v1.0.0{'\n'}
            개인화된 AI 뉴스 서비스
          </Text>
        </View>
      </ScrollView>
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
  profileContent: {
    padding: 20,
  },
  profileHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  profileInfo: {
    marginLeft: 16,
    flex: 1,
  },
  userName: {
    fontWeight: '600',
    marginBottom: 4,
  },
  interestsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    marginTop: 8,
    gap: 4,
  },
  interestChip: {
    marginRight: 4,
    marginBottom: 4,
  },
  sectionTitle: {
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 8,
    fontWeight: 'bold',
  },
  statsCard: {
    marginHorizontal: 16,
    marginBottom: 8,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontWeight: 'bold',
    marginBottom: 4,
  },
  qualitySection: {
    marginTop: 8,
  },
  qualityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  qualityBar: {
    height: 8,
    borderRadius: 4,
  },
  preferencesCard: {
    marginHorizontal: 16,
    marginBottom: 8,
  },
  activityCard: {
    marginHorizontal: 16,
    marginBottom: 8,
  },
  activityItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(0,0,0,0.1)',
  },
  activityInfo: {
    flex: 1,
    marginRight: 12,
  },
  activityStatus: {
    alignSelf: 'flex-start',
  },
  emptyText: {
    textAlign: 'center',
    padding: 20,
  },
  appInfo: {
    padding: 20,
    marginTop: 16,
  },
}); 