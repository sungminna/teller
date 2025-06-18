import React from 'react';
import { View, StyleSheet, TouchableOpacity } from 'react-native';
import {
  Card,
  Text,
  Chip,
  IconButton,
  ProgressBar,
  useTheme,
} from 'react-native-paper';
import { format } from 'date-fns';
import { ko } from 'date-fns/locale';

import { NewsItem } from '../store/newsStore';

interface NewsCardProps {
  news: NewsItem;
  onPress: () => void;
  onBookmarkToggle: () => void;
  isProcessing?: boolean;
  showPersonalizedScore?: boolean;
  personalizedScore?: number;
}

export default function NewsCard({
  news,
  onPress,
  onBookmarkToggle,
  isProcessing = false,
  showPersonalizedScore = false,
  personalizedScore = 0,
}: NewsCardProps) {
  const theme = useTheme();

  const getStatusColor = () => {
    switch (news.processingStatus) {
      case 'completed':
        return theme.colors.primary;
      case 'processing':
        return theme.colors.tertiary;
      case 'failed':
        return theme.colors.error;
      default:
        return theme.colors.outline;
    }
  };

  const getStatusText = () => {
    switch (news.processingStatus) {
      case 'completed':
        return '재생 가능';
      case 'processing':
        return '처리 중...';
      case 'failed':
        return '처리 실패';
      default:
        return '처리 대기';
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card style={styles.card} mode="elevated">
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        <Card.Content style={styles.content}>
          {/* Header */}
          <View style={styles.header}>
            <Chip
              style={[styles.categoryChip, { backgroundColor: theme.colors.primaryContainer }]}
              textStyle={{ color: theme.colors.onPrimaryContainer }}
            >
              {news.category}
            </Chip>
            <IconButton
              icon={news.isBookmarked ? 'bookmark' : 'bookmark-outline'}
              iconColor={news.isBookmarked ? theme.colors.primary : theme.colors.onSurfaceVariant}
              size={20}
              onPress={onBookmarkToggle}
              style={styles.bookmarkButton}
            />
          </View>

          {/* Title */}
          <Text
            variant="titleMedium"
            style={styles.title}
            numberOfLines={2}
            ellipsizeMode="tail"
          >
            {news.title}
          </Text>

          {/* Summary */}
          <Text
            variant="bodySmall"
            style={[styles.summary, { color: theme.colors.onSurfaceVariant }]}
            numberOfLines={3}
            ellipsizeMode="tail"
          >
            {news.summary}
          </Text>

          {/* Footer */}
          <View style={styles.footer}>
            <View style={styles.leftFooter}>
              <Chip
                style={[
                  styles.statusChip,
                  { backgroundColor: getStatusColor() + '20' }
                ]}
                textStyle={{ color: getStatusColor(), fontSize: 12 }}
                compact
              >
                {getStatusText()}
              </Chip>
              
              {news.duration && (
                <Text
                  variant="bodySmall"
                  style={[styles.duration, { color: theme.colors.onSurfaceVariant }]}
                >
                  {formatDuration(news.duration)}
                </Text>
              )}
            </View>

            <Text
              variant="bodySmall"
              style={[styles.publishedAt, { color: theme.colors.onSurfaceVariant }]}
            >
              {format(new Date(news.publishedAt), 'MM/dd HH:mm', { locale: ko })}
            </Text>
          </View>

          {/* Processing Progress */}
          {(news.processingStatus === 'processing' || isProcessing) && (
            <ProgressBar
              indeterminate
              style={styles.progressBar}
              color={theme.colors.primary}
            />
          )}

          {/* Quality Score */}
          {news.quality > 0 && (
            <View style={styles.qualityContainer}>
              <Text
                variant="bodySmall"
                style={[styles.qualityText, { color: theme.colors.onSurfaceVariant }]}
              >
                품질: {Math.round(news.quality * 100)}%
              </Text>
            </View>
          )}

          {/* Personalized Score */}
          {showPersonalizedScore && personalizedScore > 0 && (
            <View style={styles.personalizedContainer}>
              <Text
                variant="bodySmall"
                style={[styles.personalizedText, { color: theme.colors.primary }]}
              >
                🎯 맞춤도: {Math.round(personalizedScore * 100)}%
              </Text>
            </View>
          )}

          {/* Offline Indicator */}
          {news.isDownloaded && (
            <View style={styles.offlineIndicator}>
              <IconButton
                icon="download"
                iconColor={theme.colors.primary}
                size={16}
              />
              <Text
                variant="bodySmall"
                style={{ color: theme.colors.primary }}
              >
                오프라인 사용 가능
              </Text>
            </View>
          )}
        </Card.Content>
      </TouchableOpacity>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    marginVertical: 4,
  },
  content: {
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  categoryChip: {
    alignSelf: 'flex-start',
  },
  bookmarkButton: {
    margin: 0,
  },
  title: {
    marginBottom: 8,
    lineHeight: 22,
    fontWeight: '600',
  },
  summary: {
    marginBottom: 12,
    lineHeight: 18,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  leftFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  statusChip: {
    marginRight: 8,
  },
  duration: {
    marginRight: 8,
  },
  publishedAt: {
    fontSize: 12,
  },
  progressBar: {
    marginVertical: 8,
    height: 2,
  },
  qualityContainer: {
    marginTop: 4,
  },
  qualityText: {
    fontSize: 12,
  },
  personalizedContainer: {
    marginTop: 4,
  },
  personalizedText: {
    fontWeight: '500',
    fontSize: 11,
  },
  offlineIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
}); 