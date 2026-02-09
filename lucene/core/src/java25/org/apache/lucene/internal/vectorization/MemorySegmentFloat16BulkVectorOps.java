/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.internal.vectorization;

import static org.apache.lucene.internal.vectorization.PanamaVectorConstants.PREFERRED_VECTOR_BITSIZE;
import static org.apache.lucene.internal.vectorization.PanamaVectorUtilSupport.fma;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.Float16;
import jdk.incubator.vector.Float16Vector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/** Implementations of bulk vector comparison operations. Currently only supports float32. */
public final class MemorySegmentFloat16BulkVectorOps {

  static final VectorSpecies<Float16> FLOAT16_SPECIES =
      VectorSpecies.of(Float16.class, VectorShape.forBitSize(PREFERRED_VECTOR_BITSIZE));
  static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
  static final ValueLayout.OfShort LAYOUT_LE_FLOAT16 =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(LE);

  public static final DotProduct DOT_INSTANCE = new DotProduct();
  public static final Cosine COS_INSTANCE = new Cosine();
  public static final SqrDistance SQR_INSTANCE = new SqrDistance();

  private MemorySegmentFloat16BulkVectorOps() {}

  public static final class DotProduct {

    private DotProduct() {}

    public void dotProductBulk(
        MemorySegment dataSeg,
        float[] scores,
        short[] q,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      dotProductBulkImpl(dataSeg, scores, q, -1L, d1, d2, d3, d4, elementCount);
    }

    public void dotProductBulk(
        MemorySegment seg,
        float[] scores,
        long q,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      dotProductBulkImpl(seg, scores, null, q, d1, d2, d3, d4, elementCount);
    }

    public float dotProduct(MemorySegment seg, long q, long d, int elementCount) {
      int i = 0;
      Float16Vector sv = Float16Vector.zero(FLOAT16_SPECIES);
      final int limit = FLOAT16_SPECIES.loopBound(elementCount);
      for (; i < limit; i += FLOAT16_SPECIES.length()) {
        final long offset = (long) i * Short.BYTES;
        Float16Vector qv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, q + offset, LE);
        Float16Vector dv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d + offset, LE);
        sv = fma(qv, dv, sv);
      }
      short sum = sv.reduceLanes(VectorOperators.ADD);

      for (; i < elementCount; i++) {
        final long offset = (long) i * Short.BYTES;
        sum =
            fma(
                seg.get(LAYOUT_LE_FLOAT16, q + offset),
                seg.get(LAYOUT_LE_FLOAT16, d + offset),
                sum);
      }
      return Float.float16ToFloat(sum);
    }

    private void dotProductBulkImpl(
        MemorySegment seg,
        float[] scores,
        short[] qArray,
        long qOffset,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      int i = 0;
      Float16Vector sv1 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv2 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv3 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv4 = Float16Vector.zero(FLOAT16_SPECIES);

      final int limit = FLOAT16_SPECIES.loopBound(elementCount);
      for (; i < limit; i += FLOAT16_SPECIES.length()) {
        final long offset = (long) i * Short.BYTES;
        Float16Vector dv1 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d1 + offset, LE);
        Float16Vector dv2 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d2 + offset, LE);
        Float16Vector dv3 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d3 + offset, LE);
        Float16Vector dv4 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d4 + offset, LE);
        Float16Vector qv;
        if (qOffset == -1L) {
          qv = Float16Vector.fromArray(FLOAT16_SPECIES, qArray, i);
        } else {
          qv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, qOffset + offset, LE);
        }
        sv1 = fma(qv, dv1, sv1);
        sv2 = fma(qv, dv2, sv2);
        sv3 = fma(qv, dv3, sv3);
        sv4 = fma(qv, dv4, sv4);
      }
      short sum1 = sv1.reduceLanes(VectorOperators.ADD);
      short sum2 = sv2.reduceLanes(VectorOperators.ADD);
      short sum3 = sv3.reduceLanes(VectorOperators.ADD);
      short sum4 = sv4.reduceLanes(VectorOperators.ADD);

      for (; i < elementCount; i++) {
        final long offset = (long) i * Short.BYTES;
        short qValue;
        if (qOffset == -1L) {
          qValue = qArray[i];
        } else {
          qValue = seg.get(LAYOUT_LE_FLOAT16, qOffset + offset);
        }
        sum1 = fma(qValue, seg.get(LAYOUT_LE_FLOAT16, d1 + offset), sum1);
        sum2 = fma(qValue, seg.get(LAYOUT_LE_FLOAT16, d2 + offset), sum2);
        sum3 = fma(qValue, seg.get(LAYOUT_LE_FLOAT16, d3 + offset), sum3);
        sum4 = fma(qValue, seg.get(LAYOUT_LE_FLOAT16, d4 + offset), sum4);
      }
      scores[0] = Float.float16ToFloat(sum1);
      scores[1] = Float.float16ToFloat(sum2);
      scores[2] = Float.float16ToFloat(sum3);
      scores[3] = Float.float16ToFloat(sum4);
    }
  }

  // -- cosine

  public static final class Cosine {

    private Cosine() {}

    public void cosineBulk(
        MemorySegment dataSeg,
        float[] scores,
        short[] q,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      cosineBulkImpl(dataSeg, scores, q, -1L, d1, d2, d3, d4, elementCount);
    }

    public void cosineBulk(
        MemorySegment seg,
        float[] scores,
        long q,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      cosineBulkImpl(seg, scores, null, q, d1, d2, d3, d4, elementCount);
    }

    public float cosine(MemorySegment seg, long q, long d, int elementCount) {
      int i = 0;
      Float16Vector sv = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector qvNorm = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector dvNorm = Float16Vector.zero(FLOAT16_SPECIES);
      final int limit = FLOAT16_SPECIES.loopBound(elementCount);
      for (; i < limit; i += FLOAT16_SPECIES.length()) {
        final long offset = (long) i * Short.BYTES;
        Float16Vector qv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, q + offset, LE);
        Float16Vector dv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d + offset, LE);
        sv = fma(qv, dv, sv);
        qvNorm = fma(qv, qv, qvNorm);
        dvNorm = fma(dv, dv, dvNorm);
      }
      float sum = Float.float16ToFloat(sv.reduceLanes(VectorOperators.ADD));
      float qNorm = Float.float16ToFloat(qvNorm.reduceLanes(VectorOperators.ADD));
      float dNorm = Float.float16ToFloat(dvNorm.reduceLanes(VectorOperators.ADD));

      for (; i < elementCount; i++) {
        final long offset = (long) i * Short.BYTES;
        float qValue = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, q + offset));
        float dValue = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d + offset));
        sum += qValue * dValue;
        qNorm += qValue * qValue;
        dNorm += dValue * dValue;
      }
      return (float) (sum / Math.sqrt((double) qNorm * (double) dNorm));
    }

    private void cosineBulkImpl(
        MemorySegment seg,
        float[] scores,
        short[] qArray,
        long qOffset,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      int i = 0;
      Float16Vector sv1 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv2 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv3 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv4 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector qvNorm = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector dv1Norm = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector dv2Norm = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector dv3Norm = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector dv4Norm = Float16Vector.zero(FLOAT16_SPECIES);

      final int limit = FLOAT16_SPECIES.loopBound(elementCount);
      for (; i < limit; i += FLOAT16_SPECIES.length()) {
        final long offset = (long) i * Short.BYTES;
        Float16Vector dv1 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d1 + offset, LE);
        Float16Vector dv2 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d2 + offset, LE);
        Float16Vector dv3 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d3 + offset, LE);
        Float16Vector dv4 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d4 + offset, LE);
        Float16Vector qv;
        if (qOffset == -1L) {
          qv = Float16Vector.fromArray(FLOAT16_SPECIES, qArray, i);
        } else {
          qv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, qOffset + offset, LE);
        }
        qvNorm = fma(qv, qv, qvNorm);
        dv1Norm = fma(dv1, dv1, dv1Norm);
        sv1 = fma(qv, dv1, sv1);
        dv2Norm = fma(dv2, dv2, dv2Norm);
        sv2 = fma(qv, dv2, sv2);
        dv3Norm = fma(dv3, dv3, dv3Norm);
        sv3 = fma(qv, dv3, sv3);
        dv4Norm = fma(dv4, dv4, dv4Norm);
        sv4 = fma(qv, dv4, sv4);
      }
      float sum1 = Float.float16ToFloat(sv1.reduceLanes(VectorOperators.ADD));
      float sum2 = Float.float16ToFloat(sv2.reduceLanes(VectorOperators.ADD));
      float sum3 = Float.float16ToFloat(sv3.reduceLanes(VectorOperators.ADD));
      float sum4 = Float.float16ToFloat(sv4.reduceLanes(VectorOperators.ADD));
      float qNorm = Float.float16ToFloat(qvNorm.reduceLanes(VectorOperators.ADD));
      float d1N = Float.float16ToFloat(dv1Norm.reduceLanes(VectorOperators.ADD));
      float d2N = Float.float16ToFloat(dv2Norm.reduceLanes(VectorOperators.ADD));
      float d3N = Float.float16ToFloat(dv3Norm.reduceLanes(VectorOperators.ADD));
      float d4N = Float.float16ToFloat(dv4Norm.reduceLanes(VectorOperators.ADD));

      for (; i < elementCount; i++) {
        final long offset = (long) i * Short.BYTES;
        float qValue;
        if (qOffset == -1L) {
          qValue = Float.float16ToFloat(qArray[i]);
        } else {
          qValue = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, qOffset + offset));
        }
        float d1Value = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d1 + offset));
        float d2Value = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d2 + offset));
        float d3Value = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d3 + offset));
        float d4Value = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d4 + offset));
        sum1 += qValue * d1Value;
        sum2 += qValue * d2Value;
        sum3 += qValue * d3Value;
        sum4 += qValue * d4Value;
        qNorm += qValue * qValue;
        d1N += d1Value * d1Value;
        d2N += d2Value * d2Value;
        d3N += d3Value * d3Value;
        d4N += d4Value * d4Value;
      }
      scores[0] = (float) (sum1 / Math.sqrt((double) qNorm * (double) d1N));
      scores[1] = (float) (sum2 / Math.sqrt((double) qNorm * (double) d2N));
      scores[2] = (float) (sum3 / Math.sqrt((double) qNorm * (double) d3N));
      scores[3] = (float) (sum4 / Math.sqrt((double) qNorm * (double) d4N));
    }
  }

  // -- square distance

  public static final class SqrDistance {

    private SqrDistance() {}

    public void sqrDistanceBulk(
        MemorySegment dataSeg,
        float[] scores,
        short[] q,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      sqrDistanceBulkImpl(dataSeg, scores, q, -1L, d1, d2, d3, d4, elementCount);
    }

    public void sqrDistanceBulk(
        MemorySegment seg,
        float[] scores,
        long q,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      sqrDistanceBulkImpl(seg, scores, null, q, d1, d2, d3, d4, elementCount);
    }

    public float sqrDistance(MemorySegment seg, long q, long d, int elementCount) {
      int i = 0;
      Float16Vector sv = Float16Vector.zero(FLOAT16_SPECIES);
      final int limit = FLOAT16_SPECIES.loopBound(elementCount);
      for (; i < limit; i += FLOAT16_SPECIES.length()) {
        final long offset = (long) i * Short.BYTES;
        Float16Vector qv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, q + offset, LE);
        Float16Vector dv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d + offset, LE);
        Float16Vector diff = qv.sub(dv);
        sv = fma(diff, diff, sv);
      }
      float score = Float.float16ToFloat(sv.reduceLanes(VectorOperators.ADD));

      for (; i < elementCount; i++) {
        final long offset = (long) i * Short.BYTES;
        float diff =
            Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, q + offset))
                - Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d + offset));
        score += diff * diff;
      }
      return score;
    }

    private void sqrDistanceBulkImpl(
        MemorySegment seg,
        float[] scores,
        short[] qArray,
        long qOffset,
        long d1,
        long d2,
        long d3,
        long d4,
        int elementCount) {
      int i = 0;
      Float16Vector sv1 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv2 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv3 = Float16Vector.zero(FLOAT16_SPECIES);
      Float16Vector sv4 = Float16Vector.zero(FLOAT16_SPECIES);

      final int limit = FLOAT16_SPECIES.loopBound(elementCount);
      for (; i < limit; i += FLOAT16_SPECIES.length()) {
        final long offset = (long) i * Short.BYTES;
        Float16Vector dv1 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d1 + offset, LE);
        Float16Vector dv2 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d2 + offset, LE);
        Float16Vector dv3 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d3 + offset, LE);
        Float16Vector dv4 = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, d4 + offset, LE);
        Float16Vector qv;
        if (qOffset == -1L) {
          qv = Float16Vector.fromArray(FLOAT16_SPECIES, qArray, i);
        } else {
          qv = Float16Vector.fromMemorySegment(FLOAT16_SPECIES, seg, qOffset + offset, LE);
        }
        Float16Vector diff1 = qv.sub(dv1);
        Float16Vector diff2 = qv.sub(dv2);
        Float16Vector diff3 = qv.sub(dv3);
        Float16Vector diff4 = qv.sub(dv4);
        sv1 = fma(diff1, diff1, sv1);
        sv2 = fma(diff2, diff2, sv2);
        sv3 = fma(diff3, diff3, sv3);
        sv4 = fma(diff4, diff4, sv4);
      }
      float sum1 = Float.float16ToFloat(sv1.reduceLanes(VectorOperators.ADD));
      float sum2 = Float.float16ToFloat(sv2.reduceLanes(VectorOperators.ADD));
      float sum3 = Float.float16ToFloat(sv3.reduceLanes(VectorOperators.ADD));
      float sum4 = Float.float16ToFloat(sv4.reduceLanes(VectorOperators.ADD));

      for (; i < elementCount; i++) {
        final long offset = (long) i * Short.BYTES;
        float qValue;
        if (qOffset == -1L) {
          qValue = Float.float16ToFloat(qArray[i]);
        } else {
          qValue = Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, qOffset + offset));
        }
        float diff1 = qValue - Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d1 + offset));
        float diff2 = qValue - Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d2 + offset));
        float diff3 = qValue - Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d3 + offset));
        float diff4 = qValue - Float.float16ToFloat(seg.get(LAYOUT_LE_FLOAT16, d4 + offset));
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum3 += diff3 * diff3;
        sum4 += diff4 * diff4;
      }
      scores[0] = sum1;
      scores[1] = sum2;
      scores[2] = sum3;
      scores[3] = sum4;
    }
  }
}
