"""hand_tracking.model: shapes, masking, causality, learning, attention and the calibration head."""
import unittest
from model_helpers import asynchronous, randomize_heads, small, synthetic_events
import torch
from hand_tracking.model import CrossAttention, HandTransformer
from hand_tracking.objectives import calibration_loss, error_totals, pose_loss


def static_inputs(seed):
    torch.manual_seed(seed)
    base = torch.randn(2,21,3)*.1
    return base, list(synthetic_events(lambda t: base, *asynchronous()))


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_invalid_values_padding_and_backward(self):
        _, inputs = static_inputs(7)
        model = HandTransformer(small()).eval()
        events = inputs[0].shape[1]
        inputs[1] = torch.rand(1,events,2,21) > .3
        query = torch.tensor([[-.1,-.05,0.]])
        output = model(*inputs, query, return_details=True)
        self.assertEqual(output.pose.shape, (1,3,2,21,3))
        self.assertEqual(output.calibration.shape, (1,events,3,6))
        self.assertTrue(output.accepted.all())
        self.assertTrue(torch.isfinite(output.pose).all())
        altered = [value.clone() for value in inputs]
        altered[0][~inputs[1]] = float('nan')
        torch.testing.assert_close(output.pose, model(*altered, query))
        # Padding events (present False) are ignored whatever they contain.
        padded = [torch.cat((value, value[:,:4]), 1) for value in inputs]
        padded[0][:,-4:] = float('nan')
        padded[5][:,-4:] = False
        torch.testing.assert_close(output.pose, model(*padded, query), atol=1e-5, rtol=1e-5)
        self.assertFalse(model(*padded, query, return_details=True).accepted[:,-4:].any())
        target = torch.randn_like(output.pose)
        pose_loss(output.pose, target, torch.ones(1,3,2,21, dtype=torch.bool), torch.tensor([30.])).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_small_batch_can_overfit_and_metric_units(self):
        base, inputs = static_inputs(1)
        model = HandTransformer(small())
        query = torch.tensor([[-.1,-.05,0.]])
        target = (base+.05)[None,None].expand(1,3,-1,-1,-1)
        valid = torch.ones(1,3,2,21, dtype=torch.bool)
        unit = torch.tensor([30.])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = pose_loss(model(*inputs, query), target, valid, unit)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0]*.4)
        target = torch.zeros(1,2,21,3)
        prediction = target.clone()
        prediction[...,0] = .1
        error, correct, count = error_totals(prediction, target, valid[:,-1], unit)
        self.assertAlmostEqual((error/count).item(), 30., places=4)
        self.assertEqual(correct.item(), 0)

    def test_queries_use_only_events_arrived_by_then(self):
        _, inputs = static_inputs(3)
        model = randomize_heads(HandTransformer(small()).eval())
        arrival = inputs[4][0]
        query = torch.tensor([[float(arrival[-4]), float(arrival[-1])]])
        altered = [value.clone() for value in inputs]
        altered[0][0,-3:,...,5:8] = torch.nn.functional.normalize(torch.randn(3,2,21,3), dim=-1)
        with torch.no_grad():
            a, b = model(*inputs, query), model(*altered, query)
        torch.testing.assert_close(a[:,0], b[:,0])                    # later arrivals cannot change an earlier query
        self.assertFalse(torch.allclose(a[:,1], b[:,1]))


class AttentionTests(unittest.TestCase):
    def test_null_key_equals_explicit_prepended_token(self):
        torch.manual_seed(3)
        attention = CrossAttention(16, 4, 0.).eval()
        x, keys, null = torch.randn(2,5,16), torch.randn(2,7,16), torch.randn(16)
        allowed = torch.rand(2,5,7) > .5
        allowed[0,0] = False                                          # a row with only the null key
        bias = torch.randn(2,4,5,7)
        actual = attention(x, keys, allowed, bias, null=null)
        explicit = attention(x, torch.cat((null.expand(2,1,16), keys), 1),
                             torch.cat((torch.ones(2,5,1, dtype=torch.bool), allowed), -1),
                             torch.cat((torch.zeros(2,4,5,1), bias), -1))
        torch.testing.assert_close(actual, explicit, atol=1e-6, rtol=1e-6)

    def test_matches_scaled_dot_product_attention(self):
        torch.manual_seed(4)
        attention = CrossAttention(16, 4, 0.).eval()
        x, keys = torch.randn(3,5,16), torch.randn(3,7,16)
        allowed = torch.rand(3,5,7) > .4
        allowed[...,0] = True
        bias = torch.randn(3,4,5,7)
        k, v = (value.transpose(1,2) for value in attention.project_keys(keys))
        q = attention.query(x).reshape(3,5,4,4).transpose(1,2)
        mask = torch.zeros(3,1,5,7).masked_fill(~allowed[:,None], float('-inf'))+bias
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        reference = attention.out(reference.transpose(1,2).reshape(3,5,16))
        torch.testing.assert_close(attention(x, keys, allowed, bias), reference, atol=1e-6, rtol=1e-5)


class CalibrationHeadTests(unittest.TestCase):
    def test_trained_by_pose_and_aux_losses(self):
        base, inputs = static_inputs(2)
        model = HandTransformer(small())
        for p in model.calibration_output[-1].parameters():
            torch.nn.init.normal_(p, std=.01)
        output = model(*inputs, torch.tensor([[0.]]), return_details=True)
        target = torch.full((3,6), float('nan'))
        target[0] = .01
        loss = pose_loss(output.pose, (base+.02)[None,None], torch.ones(1,1,2,21, dtype=torch.bool), torch.tensor([30.]))
        loss = loss+calibration_loss(output.calibration[0], target.expand(output.calibration.shape[1],-1,-1))
        loss.backward()
        for module in (model.calibration_output[-1], model.evidence_projection[0]):
            grad = module.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_looks_only_at_the_calibration_span(self):
        _, inputs = static_inputs(6)
        model = randomize_heads(HandTransformer(small(calibration_span_s=.2)).eval())
        arrival = inputs[4][0]
        altered = [value.clone() for value in inputs]
        altered[0][0,0,...,5:8] = torch.nn.functional.normalize(torch.randn(2,21,3), dim=-1)
        with torch.no_grad():
            a = model(*inputs, torch.tensor([[0.]]), return_details=True).calibration
            b = model(*altered, torch.tensor([[0.]]), return_details=True).calibration
        # Beyond span + sample age of the altered event: unaffected.
        late = arrival > arrival[0]+.2+.2
        self.assertTrue(late.any())
        torch.testing.assert_close(a[:,late], b[:,late])
        self.assertFalse(torch.allclose(a[:,~late], b[:,~late]))

    def test_nominal_calibration_without_evidence(self):
        _, inputs = static_inputs(9)
        model = randomize_heads(HandTransformer(small()).eval())
        with torch.no_grad():
            full = model(*inputs, torch.tensor([[0.]]), return_details=True).calibration
            self.assertTrue((full[:,-1] != 0).all())
            # Only camera 0 seen: a camera alone has no relative-calibration evidence.
            alone = [value.clone() for value in inputs]
            alone[5] = inputs[2] == 0
            none = model(*alone, torch.tensor([[0.]]), return_details=True).calibration
            self.assertTrue((none == 0).all())


if __name__ == '__main__':
    unittest.main()
