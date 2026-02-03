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
}
